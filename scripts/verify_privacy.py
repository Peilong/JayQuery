#!/usr/bin/env python3
"""Audit the privacy guarantee of completed runs — turns an asserted epsilon into a verified one.

Owner: Alex Pellet

WHY THIS EXISTS
---------------
Every experiment SETS a target epsilon and solves sigma from it, so no result in the repo
demonstrates that the mechanism actually delivered the claimed guarantee. This script checks
the claim against the evidence each run recorded, and prints a table you can put in a paper
appendix. See docs/privacy_presentation_plan.md section 2.3 for the checks' rationale.

  V1  accountant cross-check   epsilon recomputed with an INDEPENDENT library (Opacus RDP)
                               agrees with autodp, so the number is not one library's bug
  V2  sensitivity enforced     max L2 row norm of the SENT embedding <= the clip S the DP
                               proof assumes
  V3  noise realized           measured std of the injected noise == sigma * S, so the run
                               spent the noise the accountant charged it for
  V4  unit of privacy          one row per student; train/val/test disjoint (dedup is a DP
                               requirement, not data cleaning)
  V5  steps consistent         every accountant charged the number of rounds actually run.
                               An UNDER-count silently under-reports epsilon
  V6  budget respected         per-source and composed epsilon within their targets

COVERAGE (not a check — a declared limitation)
----------------------------------------------
The reported epsilon covers TRAINING releases only. generate_eval_embeddings clips but does
not noise, and takes no accountant step, so validation and test embeddings are released
without formal protection. Reported below so the limitation travels with the audit.

Usage:
    python scripts/verify_privacy.py                              # every run with a meta.json
    python scripts/verify_privacy.py --runs 'd3_*' --variants dedup_first
    python scripts/verify_privacy.py --out experiments/privacy_audit.json --wandb
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from privacy.privacy_accountant import PrivacyBudgetTracker  # noqa: E402

CKPT_DIR = Path("experiments/checkpoints")
PROCESSED = Path("data/processed")

# V2: the clip is applied in float32, so the realized max row norm lands a few ulp above S.
NORM_TOL = 1e-4
# V3: the noise std is measured on one finite batch, so it fluctuates around its target.
# +/-10% is loose enough for a ~1000-row batch and tight enough to catch a wrong sigma.
NOISE_TOL = 0.10
# V1: autodp and Opacus use different bounds; they should agree to a few percent, and
# autodp should be the CONSERVATIVE (larger) one.
ACCOUNTANT_TOL = 0.05

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


@dataclass
class Check:
    """One audit result: which run, which check, the verdict, and the evidence for it."""

    run: str
    check: str
    status: str
    detail: str

    def as_row(self) -> list[str]:
        """Render as a table row."""
        return [self.run, self.check, self.status, self.detail]


def _opacus_epsilon(sigma: float, sample_rate: float, steps: int, delta: float) -> float | None:
    """Recompute epsilon with Opacus' RDP accountant, or None if Opacus is unavailable.

    Args:
        sigma: Noise multiplier (noise_std / sensitivity).
        sample_rate: Poisson sampling rate q.
        steps: Number of composed steps.
        delta: DP delta.

    Returns:
        Epsilon under Opacus' RDP accounting, or None if the import fails.
    """
    try:
        from opacus.accountants import RDPAccountant
    except ImportError:
        return None
    acct = RDPAccountant()
    for _ in range(steps):
        acct.step(noise_multiplier=sigma, sample_rate=sample_rate)
    return float(acct.get_epsilon(delta=delta))


def _sources(mech: dict) -> dict[str, dict]:
    """Every active DP source in a run, keyed by name, as {sigma, sample_rate, steps, delta}.

    Args:
        mech: The ``mechanism`` block from a run's meta.json.

    Returns:
        Mapping of source name to its accountant primitives. Empty if no source was active.
    """
    out: dict[str, dict] = {}
    q = float(mech.get("sample_rate", 1.0))
    srv = mech.get("server_dp", {})
    if srv.get("enabled") and srv.get("noise_multiplier"):
        # The server steps once per round; the horizon it was calibrated over is the
        # authority when the run was not truncated. Steps are cross-checked in V5.
        out["server"] = {
            "sigma": float(srv["noise_multiplier"]),
            "sample_rate": q,
            "delta": float(srv["delta"]),
            "target_epsilon": float(srv["target_epsilon"]),
            "steps": None,  # filled by the caller from the run's round count
        }
    for name, prims in (mech.get("per_silo") or {}).items():
        out[name] = {
            "sigma": float(prims["sigma"]),
            "sample_rate": float(prims["sample_rate"]),
            "delta": float(prims["delta"]),
            "target_epsilon": float(prims["target_epsilon"]),
            "steps": int(prims["steps"]),
            "audit": prims,
        }
    return out


def check_run(tag: str, meta: dict) -> list[Check]:
    """Run V1, V2, V3, V5 and V6 against one run's meta.json.

    Args:
        tag: The run tag (checkpoint directory name).
        meta: Parsed meta.json.

    Returns:
        One Check per assertion; a single SKIP row if the run predates the audit fields.
    """
    mech = meta.get("mechanism")
    if mech is None:
        return [
            Check(tag, "auditable", SKIP,
                  "no `mechanism` block — run predates mechanism recording, epsilon "
                  "cannot be verified from this artifact")
        ]

    sources = _sources(mech)
    if not sources:
        return [Check(tag, "auditable", SKIP, "no DP source enabled (no-DP run)")]

    spent = meta.get("epsilon_spent") or {}
    rounds_run = int(meta.get("best_round") or 0) if meta.get("stopped_early") \
        else int(meta.get("num_rounds") or 0)
    checks: list[Check] = []

    # ── V5: every accountant charged the rounds actually run ────────────────────────
    silo_steps = {n: s["steps"] for n, s in sources.items() if s["steps"] is not None}
    if silo_steps:
        distinct = set(silo_steps.values())
        if len(distinct) > 1:
            checks.append(Check(tag, "V5 steps consistent", FAIL,
                                f"silos disagree on steps charged: {silo_steps}"))
        elif not meta.get("stopped_early") and distinct != {rounds_run}:
            checks.append(Check(tag, "V5 steps consistent", FAIL,
                                f"charged {distinct.pop()} steps for {rounds_run} rounds run "
                                "— epsilon is UNDER-reported"))
        else:
            checks.append(Check(tag, "V5 steps consistent", PASS,
                                f"{sorted(distinct)[0]} steps charged, {rounds_run} rounds run"))

    # ── V7: accountant state integrity ──────────────────────────────────────────────
    # The server now forces each silo's step count to the round it actually ran, so a lost
    # step can no longer under-report epsilon. But a mismatch still means the client's
    # persisted accountant state went missing mid-run (a respawned Ray actor restoring a
    # stale snapshot), which is worth seeing even though it is no longer harmful.
    drifted = {
        name: (prims["steps_client_reported"], prims["steps"])
        for name, prims in (mech.get("per_silo") or {}).items()
        if "steps_client_reported" in prims and prims["steps_client_reported"] != prims["steps"]
    }
    if drifted:
        checks.append(Check(
            tag, "V7 accountant state integrity", WARN,
            "client accountant state was lost and rebuilt by the server (client-reported "
            "vs server-authoritative steps): "
            + ", ".join(f"{n} {c}->{s}" for n, (c, s) in drifted.items())
            + " — epsilon is correct, but the state loss indicates actor restarts",
        ))
    elif any("steps_client_reported" in p for p in (mech.get("per_silo") or {}).values()):
        checks.append(Check(
            tag, "V7 accountant state integrity", PASS,
            "client-reported and server-authoritative step counts agree",
        ))

    for name, src in sources.items():
        steps = src["steps"] if src["steps"] is not None else rounds_run
        if steps <= 0:
            continue

        # ── V1: independent accountant ──────────────────────────────────────────────
        tracker = PrivacyBudgetTracker(
            target_epsilon=src["target_epsilon"], sigma=src["sigma"],
            sample_rate=src["sample_rate"], delta=src["delta"],
        )
        tracker.steps = steps
        eps_autodp = tracker.get_epsilon()
        eps_opacus = _opacus_epsilon(src["sigma"], src["sample_rate"], steps, src["delta"])
        if eps_opacus is None:
            checks.append(Check(tag, f"V1 cross-check [{name}]", SKIP, "opacus unavailable"))
        else:
            rel = abs(eps_autodp - eps_opacus) / max(eps_autodp, 1e-12)
            direction = "conservative" if eps_autodp >= eps_opacus else "OPTIMISTIC"
            status = PASS if (rel <= ACCOUNTANT_TOL and eps_autodp >= eps_opacus) else WARN
            checks.append(Check(
                tag, f"V1 cross-check [{name}]", status,
                f"autodp {eps_autodp:.4f} vs opacus-RDP {eps_opacus:.4f} "
                f"({rel * 100:.1f}% apart, autodp {direction})",
            ))

        # ── V6: within its own budget ───────────────────────────────────────────────
        reported = spent.get(f"privacy/{name}_epsilon")
        if reported is not None:
            over = reported > src["target_epsilon"] * (1 + 1e-6)
            checks.append(Check(
                tag, f"V6 budget [{name}]", FAIL if over else PASS,
                f"spent {reported:.4f} / target {src['target_epsilon']:.4f}",
            ))

        # ── V2 + V3: the cut layer's measured evidence ──────────────────────────────
        audit = src.get("audit") or {}
        if "row_norm_max_sent" in audit:
            s = float(audit["max_norm"])
            realized_norm = float(audit["row_norm_max_sent"])
            checks.append(Check(
                tag, f"V2 sensitivity [{name}]",
                PASS if realized_norm <= s * (1 + NORM_TOL) + NORM_TOL else FAIL,
                f"max sent row norm {realized_norm:.6f} <= S={s:.4f} "
                f"(pre-clip max {float(audit['row_norm_max_pre_clip']):.3f})",
            ))
            target_std = float(audit["noise_std_target"])
            real_std = float(audit["noise_std_realized"])
            ratio = real_std / target_std if target_std else float("nan")
            checks.append(Check(
                tag, f"V3 noise realized [{name}]",
                PASS if abs(ratio - 1.0) <= NOISE_TOL else FAIL,
                f"measured std {real_std:.4f} / target {target_std:.4f} = {ratio:.3f}",
            ))

    # ── V6 (composed) ───────────────────────────────────────────────────────────────
    combined = spent.get("privacy/combined_epsilon")
    target = meta.get("combined_target_epsilon")
    if combined is not None and target:
        checks.append(Check(
            tag, "V6 budget [combined]", FAIL if combined > target * (1 + 1e-6) else PASS,
            f"composed {combined:.4f} / cap {float(target):.4f}"
            + (" (run stopped early on the gate)" if meta.get("stopped_early") else ""),
        ))

    # ── coverage: declared, not checked ─────────────────────────────────────────────
    if not mech.get("eval_releases_noised", False):
        n_evals = max(1, rounds_run // max(1, int(mech.get("eval_every", 5))))
        checks.append(Check(
            tag, "coverage", WARN,
            f"training releases only — ~{n_evals} eval releases clipped but NOT noised "
            "and not accounted (see privacy_presentation_plan section 4)",
        ))
    return checks


def check_variant(variant: str) -> list[Check]:
    """V4: the unit of privacy is one student, and the splits are disjoint.

    DP assumes one record per individual. If a student appears k times, true sensitivity is
    k*C and the reported epsilon is invalid — which is why dedup is mandatory and SMOTE is
    forbidden (docs/action_plan.md section E-a).

    Args:
        variant: Directory name under data/processed (e.g. "dedup_first").

    Returns:
        One Check per assertion about that variant's artifacts.
    """
    vdir = PROCESSED / variant
    manifest = vdir / "manifest.json"
    if not manifest.exists():
        return [Check(variant, "V4 unit of privacy", SKIP, f"no manifest at {manifest}")]
    m = json.loads(manifest.read_text())
    checks: list[Check] = []

    n_split = int(m["n_train"]) + int(m["n_val"]) + int(m["n_test"])
    n_int = int(m["n_intersection"])
    checks.append(Check(
        variant, "V4 splits partition the cohort", PASS if n_split == n_int else FAIL,
        f"train+val+test = {n_split} vs intersection {n_int}",
    ))
    checks.append(Check(
        variant, "V4 dedup applied", PASS if m.get("dedup_rule") else FAIL,
        f"rule={m.get('dedup_rule')}, dropped {m.get('n_dropped_by_dedup')} duplicate "
        f"enrolments -> {m.get('n_students')} unique students",
    ))

    ids_path = vdir / "ordered_id_list.json"
    if ids_path.exists():
        ids = json.loads(ids_path.read_text())
        dupes = len(ids) - len(set(map(str, ids)))
        checks.append(Check(
            variant, "V4 one row per student", PASS if dupes == 0 and len(ids) == n_int else FAIL,
            f"{len(ids)} aligned ids, {dupes} duplicates, intersection {n_int}",
        ))
    else:
        checks.append(Check(variant, "V4 one row per student", SKIP, f"no {ids_path}"))
    return checks


def print_table(checks: list[Check]) -> dict[str, int]:
    """Print the audit as an aligned table and return the verdict counts.

    Args:
        checks: All Check results.

    Returns:
        Mapping of status -> count.
    """
    head = ("run", "check", "status", "detail")
    widths = [
        max(len(head[i]), max((len(c.as_row()[i]) for c in checks), default=0))
        for i in range(3)
    ]
    line = "-" * (sum(widths) + 9 + 60)
    print(line)
    print(f"{head[0]:<{widths[0]}}  {head[1]:<{widths[1]}}  {head[2]:<{widths[2]}}  {head[3]}")
    print(line)
    icon = {PASS: "PASS", FAIL: "FAIL", WARN: "WARN", SKIP: "SKIP"}
    for c in checks:
        print(
            f"{c.run:<{widths[0]}}  {c.check:<{widths[1]}}  "
            f"{icon[c.status]:<{widths[2]}}  {c.detail}"
        )
    print(line)
    counts = {s: sum(1 for c in checks if c.status == s) for s in (PASS, FAIL, WARN, SKIP)}
    print("  " + "   ".join(f"{s}: {n}" for s, n in counts.items()))
    if counts[FAIL]:
        print(f"\n  {counts[FAIL]} FAILING CHECK(S) — the reported guarantee is not verified.")
        for c in checks:
            if c.status == FAIL:
                print(f"    - [{c.run}] {c.check}: {c.detail}")
    return counts


def log_to_wandb(checks: list[Check], group: str) -> None:
    """Push the audit as a browsable wandb Table.

    Args:
        checks: All Check results.
        group: wandb group to file the audit run under.
    """
    import wandb

    run = wandb.init(
        entity="jayquery", project="jayquery-vfl", name="privacy-audit",
        group=group, job_type="verification", reinit=True,
    )
    table = wandb.Table(columns=["run", "check", "status", "detail"])
    for c in checks:
        table.add_data(*c.as_row())
    counts = {s: sum(1 for c in checks if c.status == s) for s in (PASS, FAIL, WARN, SKIP)}
    run.log({"privacy_audit": table, **{f"audit/{k.lower()}": v for k, v in counts.items()}})
    run.finish()
    print("\nlogged audit table -> wandb (privacy-audit)")


def main() -> None:
    """CLI entry point: discover runs, audit them, print/persist the result."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", default="*", help="Glob over experiments/checkpoints/.")
    parser.add_argument("--variants", nargs="+", default=["dedup_first"])
    parser.add_argument("--out", type=Path, default=Path("experiments/privacy_audit.json"))
    parser.add_argument("--wandb", action="store_true", help="Also log a wandb Table.")
    parser.add_argument("--wandb-group", default="privacy-audit")
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero on FAIL (default: report and exit 0 so a pipeline stage continues).",
    )
    args = parser.parse_args()

    if not Path("pyproject.toml").exists():
        sys.exit("Run from the repo root.")

    checks: list[Check] = []
    for variant in args.variants:
        checks.extend(check_variant(variant))

    metas = sorted(CKPT_DIR.glob(f"{args.runs}/meta.json"))
    if not metas:
        print(f"no runs matched {CKPT_DIR}/{args.runs}/meta.json")
    for meta_path in metas:
        checks.extend(check_run(meta_path.parent.name, json.loads(meta_path.read_text())))

    counts = print_table(checks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"counts": counts, "checks": [vars(c) for c in checks]}, indent=2
    ))
    print(f"\naudit -> {args.out}")

    if args.wandb:
        log_to_wandb(checks, args.wandb_group)
    if args.strict and counts[FAIL]:
        sys.exit(1)


if __name__ == "__main__":
    main()
