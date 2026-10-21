#!/usr/bin/env python3
"""Publish every result into results/ — one folder a human can open and read.

Owner: Alex Pellet

WHY THIS EXISTS
---------------
Results used to live in five places: raw JSONs, 176 checkpoint directories, a generated
markdown file, a hand-written narrative doc, and wandb. None of them was "the results". This
writes ONE self-contained folder:

    results/README.md          START HERE — headline claims, figures, caveats
    results/claims.csv         every comparison with its CI (from analyze.py)
    results/caveats.json       machine-readable caveats
    results/figures/*.png      the paper's figures (from figures.py)
    results/tables/*.{csv,tex,md}   every table, paste-ready for Overleaf
    results/privacy_audit.md   the V1-V6 guarantee audit
    results/provenance.json    git SHA, timestamp, what was actually run

results/ is gitignored and fully regenerable: it is a build output, not a source of truth.

KNOWN ISSUES ARE GENERATED, NOT REMEMBERED. The open problems (budget-truncated runs,
unverifiable epsilon, the accountant under-count, unprotected eval releases) are re-detected
from the artifacts on every run and printed in README.md. They cannot silently disappear
because nobody remembered to mention them.

Usage:
    python scripts/make_report.py                 # build results/
    python scripts/make_report.py --wandb         # also push tables to wandb
    python scripts/make_report.py --no-figures    # skip plotting (faster)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figures as figures_mod  # noqa: E402
from analyze import collect_caveats  # noqa: E402
from analyze import main as _analyze_main  # noqa: E402
from results_lib import (  # noqa: E402
    FAMILIES,
    METRICS,
    PRIMARY_METRIC,
    RESULTS,
    Family,
    attack_values,
    family_sources,
    fmt_mean_sd,
    group_attack_rows,
    load_attack_rows,
    load_audit,
    load_baselines,
    load_family,
    load_variants,
)

# meta.json path of the composed (all-sources) epsilon — the reported guarantee.
COMBINED_EPS = "epsilon_spent.privacy/combined_epsilon"


# ── table construction ───────────────────────────────────────────────────────
def family_table(family: Family) -> tuple[list[str], list[list[str]]]:
    """Summarise one experiment family as a table: one row per arm, mean ± sd across seeds.

    Args:
        family: The family definition.

    Returns:
        Tuple of (headers, rows).
    """
    groups = load_family(family)
    if not groups:
        return [], []
    key_names = next(iter(groups.values())).key_names
    headers = [*key_names, "n", "eps composed", *[label for _, label, _ in METRICS],
               "trunc", "budget-stop", "val-stop", "audit"]
    rows = []
    for group in groups.values():
        # The composed (all-sources) epsilon is the system's actual guarantee, so every
        # table carries it next to the utility it bought. "-" = no DP source was active.
        cells = [*group.key, str(group.n),
                 fmt_mean_sd(group.values(COMBINED_EPS), 3)]
        for dotted, _, dec in METRICS:
            cells.append(fmt_mean_sd(group.values(dotted), dec))
        unknown = f"+{group.n_stop_unknown}?" if group.n_stop_unknown else ""
        cells += [
            f"{group.n_truncated}/{group.n}",
            f"{group.n_budget_stopped}/{group.n}{unknown}",
            f"{group.n_val_stopped}/{group.n}",
            "unverifiable" if group.n_unauditable == group.n
            else (f"{group.n - group.n_unauditable}/{group.n} ok" if group.n_unauditable
                  else "ok"),
        ]
        rows.append(cells)
    return headers, rows


def variants_table() -> tuple[list[str], list[list[str]]]:
    """Dataset provenance, one row per prepared variant.

    Returns:
        Tuple of (headers, rows).
    """
    headers = ["variant", "rule", "students", "intersection", "train/val/test",
               "label rate", "dims A/B"]
    rows = [[
        m["variant"], str(m.get("dedup_rule")), str(m.get("n_students")),
        str(m.get("n_intersection")),
        f"{m.get('n_train')}/{m.get('n_val')}/{m.get('n_test')}",
        f"{float(m.get('label_rate', 0)):.4f}",
        f"{m.get('input_dim_a')}/{m.get('input_dim_b')}",
    ] for m in load_variants()]
    return headers, rows


def baselines_table() -> tuple[list[str], list[list[str]]]:
    """Centralized and isolated-silo baselines — the comparison point for the VFL result.

    Answers "does splitting the data across silos cost anything?", which the DP numbers
    alone cannot address.

    Returns:
        Tuple of (headers, rows). Empty when baselines have not been run.
    """
    data = load_baselines()
    results = data.get("results", {})
    if not results:
        return [], []
    # Show the metrics the VFL runs report, so the two tables are readable side by side.
    wanted = ["auc_roc", "avg_precision", "f1_macro", "mcc", "accuracy", "brier_score"]
    present, seen = [], set()
    for entry in results.values():
        for key in entry.get("metrics", {}):
            base = key.split("/")[-1]
            if base in wanted and base not in seen:
                seen.add(base)
                present.append(base)
    present.sort(key=wanted.index)

    headers = ["model", "data", "n", *present]
    rows = []
    for name, entry in sorted(results.items()):
        metrics = entry.get("metrics", {})

        def pick(metric: str, metrics: dict = metrics) -> str:
            for key, values in metrics.items():
                if key.split("/")[-1] == metric:
                    return fmt_mean_sd(values, 4)
            return "-"

        n = max((len(v) for v in metrics.values()), default=0)
        rows.append([name, entry.get("data", "-"), str(n), *[pick(m) for m in present]])
    return headers, rows


def psi_table() -> tuple[list[str], list[list[str]]]:
    """PSI verification: does the real protocol reproduce the pipeline's intersection?

    Returns:
        Tuple of (headers, rows). Empty when the audit has not been run.
    """
    audit = load_audit()
    rows = [
        [c["run"], c["check"].replace("V4 ", ""), c["status"], c["detail"]]
        for c in audit.get("checks", []) if c["check"].startswith("V4")
    ]
    return (["variant", "check", "status", "detail"], rows) if rows else ([], [])


def attack_table() -> tuple[list[str], list[list[str]]]:
    """Attack results, if a sweep has been run.

    Returns:
        Tuple of (headers, rows).
    """
    rows_in = load_attack_rows()
    if not rows_in:
        return [], []
    grouped = group_attack_rows(rows_in)
    if not grouped:
        return [], []
    # Per-attribute recovery is the headline: which protected attributes came back, and by
    # how much over the always-guess-majority floor. Reported as mean ± sd across the seeds
    # whose checkpoints were attacked.
    attr_cols = sorted(
        c for c in rows_in[0] if c.startswith("attr_") and c.endswith("_lift")
    )
    metrics = [
        ("inv_asr", "ASR", 3), ("inv_asr_majority", "floor", 3),
        ("inv_asr_lift", "lift", 3),
        ("silo_a_nrmse", "A nRMSE", 2), ("silo_b_nrmse", "B nRMSE", 2),
        *[(c, c.replace("attr_", "").replace("_lift", " lift"), 3) for c in attr_cols],
        ("label_acc", "label acc", 3), ("label_acc_majority", "label floor", 3),
    ]
    headers = ["arm", "seeds", *[label for _, label, _ in metrics], "leak"]
    rows = []
    for arm, arm_rows in grouped.items():
        cells = [arm, str(len(arm_rows))]
        for key, _, dec in metrics:
            cells.append(fmt_mean_sd(attack_values(arm_rows, key), dec))
        # A negative lift is not "anti-leakage": a real attacker who does worse than the
        # floor would just guess the modal value. Leakage is therefore max(0, lift).
        lifts = attack_values(arm_rows, "inv_asr_lift")
        cells.append(f"{max(0.0, sum(lifts) / len(lifts)):.3f}" if lifts else "-")
        rows.append(cells)
    return headers, rows


def claims_table() -> tuple[list[str], list[list[str]]]:
    """The headline comparisons, read back from claims.csv on the primary metric.

    Returns:
        Tuple of (headers, rows).
    """
    path = RESULTS / "claims.csv"
    if not path.exists():
        return [], []
    primary = PRIMARY_METRIC.split(".")[-1]
    headers = ["experiment", "group", "arm", "vs", "delta", "95% CI", "p", "sig", "caveats"]
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r["metric"] != primary:
                continue
            rows.append([
                r["family"], r["split"] or "-", r["arm"], r["baseline"],
                f"{float(r['delta']):+.4f}",
                f"[{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}]",
                f"{float(r['p_value']):.4f}",
                "yes" if r["significant"] == "True" else "no",
                "yes" if r["caveats"] else "",
            ])
    return headers, rows


# ── known issues, re-detected every run ──────────────────────────────────────
def known_issues(caveats: dict) -> list[list[str]]:
    """Re-derive the standing open problems from the artifacts.

    These are the issues that must survive between sessions. Each is detected from the
    artifacts rather than remembered, so it reappears here until it is actually fixed.
    See docs/action_plan.md section R for the full write-ups.

    Args:
        caveats: Output of analyze.collect_caveats.

    Returns:
        Rows of [issue, status, detail, reference].
    """
    audit = caveats.get("audit", {})
    counts = audit.get("counts", {})
    failures = audit.get("failures", [])
    groups = caveats.get("groups", [])

    n_early_groups = sum(1 for g in groups if g.get("n_budget_stopped"))
    early_detail = "; ".join(
        f"{g['group']} ({g['n_budget_stopped']}/{g['n']})"
        for g in groups if g.get("n_budget_stopped")
    ) or "none detected"
    n_unknown_groups = sum(1 for g in groups if g.get("n_stop_unknown"))

    n_unaud = sum(1 for g in groups if g["n_unauditable"])
    unaud_runs = counts.get("SKIP", 0)

    undercount = [f for f in failures if "steps consistent" in f["check"]]
    coverage = caveats.get("coverage", [])

    # Does federating actually buy anything? If a baseline matches or beats the no-DP VFL
    # ceiling, the architecture's motivation is in question and no amount of DP analysis
    # fixes that. Checked here so it cannot be overlooked while writing.
    baselines = load_baselines().get("results", {})
    ceiling = None
    for group in load_family(FAMILIES[0]).values():
        if group.label("variant") == "dedup_first" and group.label("arm") == "no_dp":
            vals = group.values(PRIMARY_METRIC)
            ceiling = sum(vals) / len(vals) if vals else None
    beating = []
    if ceiling is not None:
        for name, entry in baselines.items():
            vals = next((v for k, v in entry.get("metrics", {}).items()
                         if k.split("/")[-1] == "auc_roc"), None)
            if vals:
                auc = sum(vals) / len(vals)
                if auc >= ceiling:
                    beating.append(f"{name} {auc:.4f}")

    # Label inference against the (un-noised) gradient channel. Out of the threat model by
    # decision (CLAUDE.md), but the paper must say so — so it stays visible here.
    label_accs = [
        float(r["label_acc"]) for r in load_attack_rows() if r.get("label_acc") not in (None, "")
    ]
    # autodp vs Opacus-RDP disagreement: both should upper-bound the same epsilon, so a
    # large gap means the reported number rests on whichever accountant is trusted.
    v1_gaps = _v1_gaps(audit.get("checks", []) or load_audit().get("checks", []))

    return [
        [
            "Labels recoverable from gradients",
            ("OPEN (out of threat model — must be stated in the paper)" if label_accs
             else "not checked"),
            (f"label-inference accuracy {min(label_accs):.3f}-{max(label_accs):.3f} across "
             f"{len(label_accs)} attacked checkpoints, every DP config included. Gradients "
             "are not noised; labels are treated as known to the silos.")
            if label_accs else "run the attack stage",
            "CLAUDE.md threat model",
        ],
        [
            "autodp and Opacus-RDP disagree on epsilon",
            "OPEN" if v1_gaps else "clear",
            ("autodp is lower than Opacus-RDP by more than 5% on: "
             + "; ".join(f"{arm} (max {gap:.0f}%)" for arm, gap in v1_gaps.items())
             + ". Report the larger (conservative) value, or justify autodp's tighter bound.")
            if v1_gaps else "the two accountants agree within 5% on every run",
            "privacy_audit.md V1",
        ],
        [
            "Baseline matches or beats the federated model",
            "OPEN" if beating else ("clear" if baselines else "not checked"),
            (f"VFL no-DP ceiling is {ceiling:.4f}; these baselines meet or beat it: "
             + ", ".join(beating)
             + ". Not a threat to the feasibility framing, but it makes the "
               "federation-cost number load-bearing: it needs seeds and a like-for-like "
               "training budget (VFL runs are horizon-truncated).")
            if beating else ("no baseline beats the federated ceiling" if baselines
                             else "run the baselines stage"),
            "action_plan.md §R6",
        ],
        [
            "Budget-truncated runs in reported arms",
            "OPEN" if n_early_groups else ("unknown" if n_unknown_groups else "clear"),
            (f"{n_early_groups} arm(s) contain runs a PRIVACY BUDGET ended — under-trained, "
             f"not over-noised: {early_detail}." if n_early_groups else
             "no run was ended by a privacy budget.")
            + (f" {n_unknown_groups} arm(s) have runs that stopped early for an unrecorded "
               "reason (predate stop_reason); re-run to disambiguate."
               if n_unknown_groups else ""),
            "action_plan.md §R5",
        ],
        [
            "Checkpoints with no mechanism record",
            "OPEN" if (n_unaud or unaud_runs) else "clear",
            f"{unaud_runs} run(s) carry no `mechanism` block, so their epsilon cannot be "
            f"recomputed from the artifact; {n_unaud} reported arm(s) affected. "
            "Fix: re-run with `reproduce_all.sh --fresh`.",
            "action_plan.md §R4",
        ],
        [
            "Accountant under-counts a step (epsilon under-reported)",
            "OPEN" if undercount else ("clear" if counts else "not checked"),
            (f"{len(undercount)} run(s) charged fewer accountant steps than rounds run: "
             + "; ".join(f["run"] for f in undercount)) if undercount
            else ("no run failed V5 in the current audit" if counts
                  else "run scripts/verify_privacy.py to check"),
            "action_plan.md §R1",
        ],
        [
            "Validation/test embeddings released unprotected",
            "OPEN (by design — must be stated in the paper)" if coverage else "not checked",
            (f"{coverage[0]['n_runs']} run(s): eval embeddings are clipped but NOT noised and "
             "take no accountant step, so val/test students are outside the reported "
             "guarantee.") if coverage else "run scripts/verify_privacy.py to check",
            "action_plan.md §R2 · privacy_presentation_plan.md §4",
        ],
    ]


# ── rendering ────────────────────────────────────────────────────────────────
_V1_RE = re.compile(r"autodp ([\d.]+) vs opacus-RDP ([\d.]+)")


def _v1_gaps(checks: list[dict], threshold: float = 0.05) -> dict[str, float]:
    """Arms where autodp's epsilon is materially below the Opacus-RDP cross-check.

    Args:
        checks: Rows of experiments/privacy_audit.json.
        threshold: Relative gap (Opacus / autodp - 1) below which a WARN is ignored.

    Returns:
        {arm tag without the seed suffix: largest gap in percent}.
    """
    gaps: dict[str, float] = {}
    for c in checks:
        if not c["check"].startswith("V1") or c["status"] != "WARN":
            continue
        m = _V1_RE.search(c["detail"])
        if not m or float(m.group(1)) <= 0:
            continue
        gap = float(m.group(2)) / float(m.group(1)) - 1
        if gap > threshold:
            arm = re.sub(r"_s\d+$", "", c["run"])
            gaps[arm] = max(gaps.get(arm, 0.0), 100 * gap)
    return gaps


def digest(
    tables: dict[str, tuple[list[str], list[list[str]]]],
    caveats: dict,
    provenance: dict,
    audit: dict,
) -> str:
    """A single dense page with every number, for a later Claude session writing the paper.

    README.md is for a human browsing the folder (figures, links). DIGEST.md is the one
    file to read when drafting text: all tables inline, the open issues, the audit
    rolled up by check, and the statements the paper is obliged to make. Nothing here is
    interpreted beyond what the artifacts say.

    Args:
        tables: Built tables.
        caveats: Output of analyze.collect_caveats.
        provenance: The provenance dict.
        audit: Parsed privacy_audit.json.

    Returns:
        Markdown text.
    """
    git = provenance.get("git", {})
    parts = [
        "# JayQuery results digest",
        "",
        "_Machine-oriented summary: every table inline, no figures. Regenerated by "
        "`scripts/make_report.py`; do not edit._",
        "",
        "## Provenance",
        "",
        f"- generated: {provenance.get('generated_utc')}",
        f"- commit: `{(git.get('commit') or '?')[:10]}` on `{git.get('branch')}`"
        + (" (UNCOMMITTED CHANGES)" if git.get("dirty") else ""),
        "- runs: " + ", ".join(f"{k}={v}" for k, v in provenance.get("run_counts", {}).items()),
        "- audit: " + ", ".join(f"{k}={v}" for k, v in audit.get("counts", {}).items()),
        "",
        "## Open issues",
        "",
        md_table(["issue", "status", "detail", "reference"], known_issues(caveats)),
        "",
    ]

    checks = audit.get("checks", [])
    if checks:
        roll: dict[tuple[str, str], int] = {}
        for c in checks:
            key = (re.sub(r" \[.*\]$", "", c["check"]), c["status"])
            roll[key] = roll.get(key, 0) + 1
        parts += ["## Privacy audit by check", "",
                  md_table(["check", "status", "runs"],
                           [[k, st, str(n)] for (k, st), n in sorted(roll.items())]), ""]

    parts += [
        "## Statements the paper must make",
        "",
        "- Threat model protects silo FEATURES (embeddings on the wire + server). Labels are "
        "out of scope: gradients are not noised and label inference is ~100% (t7).",
        "- Val/test embeddings are clipped but not noised; val/test students are outside the "
        "formal guarantee.",
        "- The reported guarantee is the COMPOSED epsilon (`eps composed` column), not the "
        "per-mechanism target.",
        "- DP arms keep AUC but predictions collapse toward the base rate (`pred_std` ~0.01-0.02 "
        "vs ~0.36 without noise): ranking survives, probabilities are not calibrated.",
        "- Attack `leak` = max(0, lift); a lift below the floor is 'no better than guessing', "
        "not negative leakage.",
        "- Rows with `trunc` > 0 were still improving at the round horizon.",
        "",
        "## Tables",
        "",
    ]
    for name, (headers, rows) in tables.items():
        parts += [f"### {name}", "", md_table(headers, rows), ""]
    return "\n".join(parts)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table.

    Args:
        headers: Column headers.
        rows: Stringified cells.

    Returns:
        Markdown text.
    """
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_table(out_dir: Path, name: str, headers: list[str], rows: list[list[str]]) -> None:
    """Write one table as CSV, LaTeX and markdown.

    LaTeX is emitted alongside CSV so a table can go straight into Overleaf without being
    retyped — retyping is how a paper number drifts from the artifact that produced it.

    Args:
        out_dir: Tables directory.
        name: File stem.
        headers: Column headers.
        rows: Stringified cells.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    (out_dir / f"{name}.md").write_text(md_table(headers, rows) + "\n")

    def esc(s: str) -> str:
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")

    tex = [
        r"\begin{tabular}{" + "l" * len(headers) + "}", r"\toprule",
        " & ".join(esc(h) for h in headers) + r" \\", r"\midrule",
        *[" & ".join(esc(c) for c in r) + r" \\" for r in rows],
        r"\bottomrule", r"\end{tabular}",
    ]
    (out_dir / f"{name}.tex").write_text("\n".join(tex) + "\n")


def git_provenance() -> dict:
    """Repository state, so a result can be traced to the code that produced it.

    Returns:
        Dict with the commit SHA, branch and dirty flag (values are None if git fails).
    """
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def build(out_dir: Path, do_figures: bool = True) -> dict[str, tuple[list[str], list[list[str]]]]:
    """Build the whole results/ folder.

    Args:
        out_dir: Root of the results directory.
        do_figures: Whether to draw the figures.

    Returns:
        {table name: (headers, rows)} for optional wandb logging.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    # Clear generated subdirs first. Table names carry a positional prefix, so adding an
    # experiment renumbers them (t3_x -> t4_x) and the old file would survive alongside
    # the new one — two tables, same content, different numbers, one of them stale.
    for stale_dir in (tables_dir, out_dir / "figures"):
        if stale_dir.exists():
            for old in stale_dir.iterdir():
                if old.is_file():
                    old.unlink()
    tables: dict[str, tuple[list[str], list[list[str]]]] = {}

    # 1. claims + caveats (analyze.py writes claims.csv / caveats.json into out_dir)
    sys.argv = ["analyze", "--out-dir", str(out_dir)]
    _analyze_main()
    caveats = collect_caveats()

    # 2. figures
    written_figs: list[str] = []
    if do_figures:
        print("\nfigures:")
        written_figs = figures_mod.build_all(out_dir / "figures")

    # 3. tables
    def add(name: str, headers: list[str], rows: list[list[str]]) -> None:
        if rows:
            tables[name] = (headers, rows)
            write_table(tables_dir, name, headers, rows)

    add("t0_dataset_variants", *variants_table())
    add("t1_headline_claims", *claims_table())
    add("t1b_baselines", *baselines_table())
    for i, family in enumerate(FAMILIES, start=2):
        add(f"t{i}_{family.name}", *family_table(family))
    add("t6_psi_verification", *psi_table())
    add("t7_attack_results", *attack_table())

    # 4. privacy audit as its own readable page
    audit = load_audit()
    if audit:
        lines = ["# Privacy audit", "",
                 "Generated by `scripts/verify_privacy.py`. Each row checks the reported "
                 "guarantee against evidence the run recorded about itself.", "",
                 "  ".join(f"**{k}**: {v}" for k, v in audit.get("counts", {}).items()), ""]
        notable = [c for c in audit.get("checks", []) if c["status"] in ("FAIL", "WARN")]
        if notable:
            lines += ["## Non-passing checks", "",
                      md_table(["run", "check", "status", "detail"],
                               [[c["run"], c["check"], c["status"], c["detail"]]
                                for c in notable]), ""]
        else:
            lines += ["_All checks passed._", ""]
        (out_dir / "privacy_audit.md").write_text("\n".join(lines) + "\n")

    # 5. provenance
    provenance = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_provenance(),
        "sources": {
            f.name: [str(p) for p in family_sources(f)] for f in FAMILIES
        },
        "run_counts": {
            f.name: sum(g.n for g in load_family(f).values()) for f in FAMILIES
        },
        "audit_counts": audit.get("counts", {}),
        "note": "Run mode (simulation vs deployment) is not recorded in meta.json; "
                "provenance for that currently comes from the run tag and the matrix job list.",
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    # 6. DIGEST — every number on one page, for drafting the paper
    (out_dir / "DIGEST.md").write_text(digest(tables, caveats, provenance, audit) + "\n")

    # 7. README — the entry point
    (out_dir / "README.md").write_text(
        readme(tables, caveats, provenance, written_figs) + "\n"
    )
    return tables


def readme(
    tables: dict[str, tuple[list[str], list[list[str]]]],
    caveats: dict,
    provenance: dict,
    figs: list[str],
) -> str:
    """Compose results/README.md — the one page to open first.

    Args:
        tables: Built tables.
        caveats: Output of collect_caveats.
        provenance: Provenance record.
        figs: Figure filenames that were written.

    Returns:
        Markdown text.
    """
    git = provenance["git"]
    dirty = " (uncommitted changes present)" if git.get("dirty") else ""
    parts = [
        "# JayQuery — Results",
        "",
        f"_Generated {provenance['generated_utc']} from commit "
        f"`{(git.get('commit') or 'unknown')[:10]}` on `{git.get('branch')}`{dirty}._",
        "",
        "Everything here is a build output of `bash scripts/reproduce_all.sh` and is "
        "regenerated from the run artifacts. Nothing in this folder is hand-written, and "
        "nothing here is committed to git.",
        "",
        "---",
        "",
        "## ⚠️ Read before quoting any number",
        "",
        "These are re-detected from the artifacts every time this report is built, so they "
        "stay visible until they are actually fixed.",
        "",
        md_table(["issue", "status", "detail", "reference"], known_issues(caveats)),
        "",
        "---",
        "",
        "## Headline claims",
        "",
    ]

    if "t1_headline_claims" in tables:
        parts += [
            "Each row is one arm compared against its own baseline with Welch's "
            "unequal-variance test across seeds. Computed by `scripts/analyze.py`; "
            "full detail including every metric is in [`claims.csv`](claims.csv).",
            "",
            md_table(*tables["t1_headline_claims"]),
            "",
        ]
    else:
        parts += ["_No claims computed — run `python scripts/analyze.py`._", ""]

    if figs:
        parts += ["## Figures", ""]
        captions = {
            "fig1_privacy_utility_curve": "Privacy-utility trade-off. Hollow markers are "
                                          "points where at least one seed stopped early "
                                          "(validation early stopping or the budget gate; "
                                          "t4 says which).",
            "fig2_epsilon_isocline": "Utility spread at an IDENTICAL formal guarantee "
                                     "(epsilon = 1.0). The clip S does not enter the epsilon "
                                     "calculation, so this spread costs no privacy.",
            "fig3_attack_vs_epsilon": "Attack success against the transmitted (clipped + "
                                      "noised) embedding, with the uninformed-attacker floor.",
            "fig4_variance_vs_epsilon": "Seed-to-seed variance against epsilon.",
            "fig5_attack_vs_epsilon": "Inversion attack success against the cut-layer "
                                      "epsilon, with the no-noise level and the "
                                      "uninformed-attacker floor.",
        }
        for fig in figs:
            stem = Path(fig).stem
            parts += [f"### {stem}", "", f"![{stem}](figures/{fig})", "",
                      f"_{captions.get(stem, '')}_", ""]

    parts += ["## Tables", "",
              "Every table is written as `.csv`, `.tex` (paste-ready for Overleaf) and "
              "`.md` under [`tables/`](tables/).", ""]
    for name, (_, rows) in tables.items():
        parts.append(f"- [`{name}`](tables/{name}.md) — {len(rows)} rows")
    parts += ["", "## Files", "",
              "| file | what it is |",
              "|---|---|",
              "| `README.md` | this page |",
              "| `DIGEST.md` | every table + open issues on one page (for drafting the paper) |",
              "| `claims.csv` | every comparison: delta, 95% CI, p, n, caveats |",
              "| `caveats.json` | machine-readable run-health and audit findings |",
              "| `privacy_audit.md` | the V1-V6 guarantee audit |",
              "| `provenance.json` | commit, timestamp, artifact sources, run counts |",
              "| `figures/` | the paper's figures |",
              "| `tables/` | every table as csv / tex / md |",
              "",
              "## How this was produced", "",
              "```bash",
              "bash scripts/reproduce_all.sh --fresh    # re-run everything, then publish",
              "bash scripts/reproduce_all.sh            # reuse artifacts, republish",
              "```", ""]

    counts = provenance.get("run_counts", {})
    parts += ["Runs behind this report: " +
              ", ".join(f"{k} = {v}" for k, v in counts.items() if v) + ".", ""]
    return "\n".join(parts)


def log_to_wandb(tables: dict[str, tuple[list[str], list[list[str]]]], group: str) -> None:
    """Push every table to wandb as a browsable panel.

    Args:
        tables: Built tables.
        group: wandb group for the report run.
    """
    import wandb

    run = wandb.init(
        entity="jayquery", project="jayquery-vfl", name="results-report",
        group=group, job_type="report", reinit=True,
    )
    payload = {}
    for name, (headers, rows) in tables.items():
        table = wandb.Table(columns=headers)
        for row in rows:
            table.add_data(*row)
        payload[f"report/{name}"] = table
    run.log(payload)
    run.finish()
    print(f"logged {len(payload)} tables -> wandb (results-report)")


def main() -> None:
    """CLI entry point: build results/ and optionally mirror the tables to wandb."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="paper-report")
    args = parser.parse_args()

    if not Path("pyproject.toml").exists():
        sys.exit("Run from the repo root.")

    tables = build(args.out_dir, do_figures=not args.no_figures)
    print(f"\nresults -> {args.out_dir}/  ({len(tables)} tables)")
    print(f"START HERE: {args.out_dir / 'README.md'}")
    if args.wandb:
        log_to_wandb(tables, args.wandb_group)


if __name__ == "__main__":
    main()
