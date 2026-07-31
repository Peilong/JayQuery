#!/usr/bin/env python3
"""Compute the paper's headline claims from the artifacts — every delta, CI and caveat.

Owner: Alex Pellet

WHY THIS EXISTS
---------------
Before this script, the paper's central numbers were HAND-TYPED. "server_only vs the no-DP
ceiling: -0.0022, 95% CI [-0.0053, +0.0009]" existed only as characters in docs/FINDINGS.md;
no code produced it, so it could not be regenerated or checked. This computes every such
comparison from the run artifacts, with Welch's unequal-variance test (DP arms carry up to
10x the seed variance of the ceiling, so a pooled test would understate the interval).

Each claim carries its own caveats, so a number that rests on truncated or unauditable runs
says so in the same row rather than in a document somebody has to remember to read:

  * runs stopped early by the privacy budget gate  -> under-trained, not over-noised
  * runs still improving at the horizon            -> the value is a floor (section K5)
  * runs with no mechanism record                  -> epsilon unverifiable (section R4)

Outputs:
    results/claims.csv     one row per (family, group, arm, metric) comparison
    results/caveats.json   machine-readable caveats, consumed by make_report.py

Usage:
    python scripts/analyze.py
    python scripts/analyze.py --metric metrics.recall_at_precision_70
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from results_lib import (  # noqa: E402
    FAMILIES,
    METRICS,
    PRIMARY_METRIC,
    RESULTS,
    Family,
    Group,
    load_audit,
    load_baselines,
    load_family,
    welch,
)

CLAIM_FIELDS = [
    "family", "split", "arm", "baseline", "metric",
    "n_arm", "n_baseline", "mean_arm", "mean_baseline",
    "delta", "ci_low", "ci_high", "se", "df", "p_value", "significant", "caveats",
]


def _compare_family(family: Family, metric_keys: list[str]) -> list[dict]:
    """Every arm-vs-baseline comparison within one experiment family.

    Comparisons are made WITHIN a split (e.g. arms are compared to the ceiling of their own
    dedup variant), never across splits — comparing an arm in one variant to a ceiling in
    another would confound the DP effect with the label-definition effect.

    Args:
        family: The family definition.
        metric_keys: Dotted metric paths to compare on.

    Returns:
        One dict per comparison, ready for CSV.
    """
    groups = load_family(family)
    if not groups:
        return []

    # Partition into {split value: {arm value: Group}}.
    splits: dict[str, dict[str, Group]] = {}
    for group in groups.values():
        split = group.label(family.split_by) if family.split_by else ""
        splits.setdefault(split, {})[group.label(family.arm_key)] = group

    rows: list[dict] = []
    for split, arms in sorted(splits.items()):
        base = arms.get(family.baseline)
        if base is None:
            continue
        for arm_name, group in arms.items():
            if arm_name == family.baseline:
                continue
            for metric in metric_keys:
                res = welch(group.values(metric), base.values(metric))
                if res is None:
                    continue
                caveats = group.caveats() + [f"baseline: {c}" for c in base.caveats()]
                rows.append({
                    "family": family.name,
                    "split": split,
                    "arm": arm_name,
                    "baseline": family.baseline,
                    "metric": metric.split(".")[-1],
                    "n_arm": group.n,
                    "n_baseline": base.n,
                    "mean_arm": round(res["mean_arm"], 6),
                    "mean_baseline": round(res["mean_baseline"], 6),
                    "delta": round(res["delta"], 6),
                    "ci_low": round(res["ci_low"], 6),
                    "ci_high": round(res["ci_high"], 6),
                    "se": round(res["se"], 6),
                    "df": round(res["df"], 2),
                    "p_value": round(res["p_value"], 6),
                    "significant": res["significant"],
                    "caveats": "; ".join(caveats),
                })
    return rows


def _compare_federation_cost(metric_keys: list[str]) -> list[dict]:
    """VFL vs the centralized baselines — the feasibility paper's headline claim.

    Under the framing "an architecture of this shape is implementable, and you lose little
    utility by doing so", the cost of federation IS the central number. It was previously
    computed by hand outside the repo, which is exactly the failure this script exists to
    prevent.

    The comparison that matters is against the ARCHITECTURE-MATCHED centralized MLP: it
    shares the model family and feature set with the VFL run, so the delta isolates the
    federated split rather than the model class. xgboost and the isolated-silo models are
    compared too, as context.

    Args:
        metric_keys: Dotted metric paths to compare on.

    Returns:
        One dict per (baseline model, metric) comparison, ready for CSV.
    """
    baselines = load_baselines().get("results", {})
    if not baselines:
        return []

    # The federated reference: the no-DP arm of the primary variant (its ceiling).
    matrix = load_family(FAMILIES[0])
    vfl = next(
        (g for g in matrix.values()
         if g.label("variant") == "dedup_first" and g.label("arm") == "no_dp"),
        None,
    )
    if vfl is None:
        return []

    rows: list[dict] = []
    for model, entry in sorted(baselines.items()):
        for metric in metric_keys:
            short = metric.split(".")[-1]
            values = next(
                (v for k, v in entry.get("metrics", {}).items() if k.split("/")[-1] == short),
                None,
            )
            if not values:
                continue
            # delta = centralized - federated, i.e. what federating COSTS (positive = cost).
            res = welch(values, vfl.values(metric))
            if res is None:
                continue
            caveats = [f"VFL side: {c}" for c in vfl.caveats()]
            caveats.append(
                "training budgets are not matched: VFL runs a fixed round horizon with "
                "Poisson batches while baselines train to convergence, so this is an "
                "UPPER BOUND on the true cost of federation (action_plan.md §R6b)"
            )
            rows.append({
                "family": "federation_cost",
                "split": "dedup_first",
                "arm": model,
                "baseline": "vfl_no_dp",
                "metric": short,
                "n_arm": len(values),
                "n_baseline": vfl.n,
                "mean_arm": round(res["mean_arm"], 6),
                "mean_baseline": round(res["mean_baseline"], 6),
                "delta": round(res["delta"], 6),
                "ci_low": round(res["ci_low"], 6),
                "ci_high": round(res["ci_high"], 6),
                "se": round(res["se"], 6),
                "df": round(res["df"], 2),
                "p_value": round(res["p_value"], 6),
                "significant": res["significant"],
                "caveats": "; ".join(caveats),
            })
    return rows


def collect_caveats() -> dict:
    """Gather every standing caveat that qualifies the results, from the artifacts.

    These are the issues that must not get lost between sessions. Each is derived from an
    artifact rather than remembered, so it reappears on every regeneration until fixed.

    Returns:
        Dict with per-group run-health caveats and privacy-audit findings.
    """
    out: dict = {"groups": [], "audit": {}, "coverage": []}

    for family in FAMILIES:
        for group in load_family(family).values():
            if cav := group.caveats():
                out["groups"].append({
                    "family": family.name,
                    "group": " / ".join(group.key),
                    "n": group.n,
                    "n_early": group.n_early,
                    "n_budget_stopped": group.n_budget_stopped,
                    "n_val_stopped": group.n_val_stopped,
                    "n_stop_unknown": group.n_stop_unknown,
                    "n_truncated": group.n_truncated,
                    "n_unauditable": group.n_unauditable,
                    "caveats": cav,
                })

    audit = load_audit()
    if audit:
        out["audit"]["counts"] = audit.get("counts", {})
        checks = audit.get("checks", [])
        out["audit"]["failures"] = [c for c in checks if c["status"] == "FAIL"]
        # The eval-coverage warning is the same sentence on every run; report it once with
        # a count rather than repeating it N times.
        coverage = [c for c in checks if c["check"] == "coverage"]
        if coverage:
            out["coverage"] = [{
                "n_runs": len(coverage),
                "detail": coverage[0]["detail"],
            }]
        out["audit"]["unauditable_runs"] = sum(
            1 for c in checks if c["status"] == "SKIP" and c["check"] == "auditable"
        )
    return out


def main() -> None:
    """CLI entry point: compute claims + caveats and write them under results/."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--metric", action="append", default=None,
        help="Dotted metric path to compare on; repeatable. Default: every reported metric.",
    )
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    if not Path("pyproject.toml").exists():
        sys.exit("Run from the repo root.")

    metric_keys = args.metric or [m[0] for m in METRICS]
    rows: list[dict] = []
    for family in FAMILIES:
        rows.extend(_compare_family(family, metric_keys))
    rows.extend(_compare_federation_cost(metric_keys))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    claims_path = args.out_dir / "claims.csv"
    with claims_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLAIM_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    caveats = collect_caveats()
    (args.out_dir / "caveats.json").write_text(json.dumps(caveats, indent=2))

    print(f"claims  -> {claims_path}  ({len(rows)} comparisons)")
    print(f"caveats -> {args.out_dir / 'caveats.json'}  "
          f"({len(caveats['groups'])} groups flagged)")

    # Echo the headline comparisons so running this alone is useful.
    primary = PRIMARY_METRIC.split(".")[-1]
    head = [r for r in rows if r["metric"] == primary and r["family"] == "utility_matrix"]
    if head:
        print(f"\nHeadline ({primary}, vs the no-DP ceiling):")
        for r in sorted(head, key=lambda r: (r["split"], r["arm"])):
            star = "*" if r["significant"] else " "
            flag = " ⚠" if r["caveats"] else "  "
            print(f"  {r['split']:<14} {r['arm']:<12} {r['delta']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]{star}{flag}")
        print("  (* = 95% CI excludes zero; ⚠ = caveats apply, see results/caveats.json)")

    fed = [r for r in rows if r["family"] == "federation_cost" and r["metric"] == primary]
    if fed:
        print(f"\nCost of federation ({primary}, centralized minus VFL — positive = cost):")
        for r in sorted(fed, key=lambda r: -r["delta"]):
            star = "*" if r["significant"] else " "
            print(f"  {r['arm']:<22} {r['delta']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]{star}")
        print("  NOTE: budgets are not matched — these are UPPER BOUNDS (§R6b).")


if __name__ == "__main__":
    main()
