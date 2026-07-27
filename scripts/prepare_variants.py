#!/usr/bin/env python3
"""Regenerate every dataset variant with one command.

Owner: Alex Pellet

Shells out to scripts/prepare_silos.py once per variant, so each dataset is built
by exactly the same code path a manual invocation uses (and a failure in one
variant cannot corrupt another). Each variant lands in its own directory with a
manifest.json recording the parameters that produced it.

The three agreed variants (docs/action_plan.md §E-a) differ ONLY in --dedup-rule:

    first  — earliest enrolment chronologically. PRIMARY result. Label-neutral
             (selects on time, not outcome) and mirrors deployment: predict on a
             student's first course with no look-ahead.
    best   — keep the student's best outcome.  ) label-SELECTING; report as
    worst  — keep the student's worst outcome. ) sensitivity bounds ONLY.

12.3% of OULAD students appear in >1 module-presentation and 33% of those have
MIXED outcomes, so the rule moves the label rate by ~4 points — hence reporting
best/worst as bounds around first.

Every variant also applies the §F pre-start-withdrawal exclusion: withdrawals that
unregistered on or before --withdrawal-cutoff-day are non-starters, not dropout
risk, and are dropped.

Point a training run at a variant with the `processed-dir` run-config key:

    flwr run . --run-config "processed-dir='data/processed/dedup_first'"

Usage:
    python scripts/prepare_variants.py                      # all 3, all modules
    python scripts/prepare_variants.py --rules first        # just the primary
    python scripts/prepare_variants.py --withdrawal-cutoff-day 14 \
        --tag cutoff14                                      # §F4 sensitivity sweep
    python scripts/prepare_variants.py --modules CCC_2014J --tag ccc
    python scripts/prepare_variants.py --dry-run            # print commands only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
DEFAULT_RULES = ("first", "best", "worst")


def variant_dir(base: Path, rule: str, tag: str | None) -> Path:
    """Output directory for one variant.

    Args:
        base: Parent directory for all variants (default data/processed).
        rule: Dedup rule name.
        tag: Optional suffix distinguishing a non-default parameter sweep
            (e.g. "cutoff14"), so it cannot overwrite the canonical variants.

    Returns:
        Path like ``data/processed/dedup_first`` or ``data/processed/dedup_first_cutoff14``.
    """
    name = f"dedup_{rule}" if not tag else f"dedup_{rule}_{tag}"
    return base / name


def build_command(
    rule: str, out_dir: Path, modules: str, cutoff_day: str, seed: int
) -> list[str]:
    """Build the prepare_silos.py argv for one variant."""
    return [
        sys.executable,
        "scripts/prepare_silos.py",
        "--modules", modules,
        "--dedup-rule", rule,
        "--withdrawal-cutoff-day", cutoff_day,
        "--seed", str(seed),
        "--out_dir", str(out_dir),
    ]


def summarize(dirs: dict[str, Path]) -> None:
    """Print a comparison table from each variant's manifest.json.

    Args:
        dirs: Mapping of rule name -> variant directory.
    """
    rows = []
    for rule, d in dirs.items():
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            rows.append((rule, "MISSING manifest.json", "", "", "", "", ""))
            continue
        m = json.loads(manifest_path.read_text())
        rows.append((
            rule,
            str(m["n_students"]),
            str(m["n_intersection"]),
            f"{m['n_train']}/{m['n_val']}/{m['n_test']}",
            f"{m['label_rate']:.4f}",
            f"{m['label_rate_test']:.4f}",
            f"{m['input_dim_a']}/{m['input_dim_b']}",
        ))

    header = ("rule", "cohort", "intersect", "train/val/test", "label", "label_test", "dimA/dimB")
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))
    print("\n" + line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)))

    labels = [
        json.loads((d / "manifest.json").read_text())["label_rate"]
        for d in dirs.values()
        if (d / "manifest.json").exists()
    ]
    if len(labels) > 1:
        print(
            f"\nlabel-rate spread across rules: {max(labels) - min(labels):.4f} "
            "— report 'first' as the primary result and best/worst as bounds."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rules",
        nargs="+",
        choices=DEFAULT_RULES,
        default=list(DEFAULT_RULES),
        help="Dedup rules to build (default: all three).",
    )
    parser.add_argument(
        "--modules",
        default="all",
        help="Passed through to prepare_silos.py --modules. Default 'all' "
        "(every module-presentation).",
    )
    parser.add_argument(
        "--withdrawal-cutoff-day",
        default="0",
        help="Passed through to prepare_silos.py. Default 0; 'none' disables the "
        "pre-start-withdrawal exclusion.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Parent directory for the variant subdirectories.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Suffix for the variant directory names, e.g. --tag cutoff14. Use this "
        "whenever you change --modules/--withdrawal-cutoff-day so a sweep cannot "
        "overwrite the canonical dedup_{first,best,worst} datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the variant directory already has a manifest.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    args = parser.parse_args()

    if not Path("scripts/prepare_silos.py").exists():
        sys.exit("Run this from the repo root (scripts/prepare_silos.py not found).")

    # Guard the canonical directory names against non-default parameters.
    non_default = args.modules != "all" or args.withdrawal_cutoff_day != "0"
    if non_default and not args.tag and not args.dry_run:
        sys.exit(
            "Refusing to write canonical dedup_* directories with non-default "
            f"parameters (modules={args.modules!r}, "
            f"withdrawal-cutoff-day={args.withdrawal_cutoff_day!r}). "
            "Pass --tag <name> to give this sweep its own directories."
        )

    dirs = {rule: variant_dir(args.base_dir, rule, args.tag) for rule in args.rules}
    failures: list[str] = []

    for rule in args.rules:
        out_dir = dirs[rule]
        cmd = build_command(
            rule, out_dir, args.modules, args.withdrawal_cutoff_day, args.seed
        )
        print(f"\n=== variant: {rule} -> {out_dir} ===")

        if args.dry_run:
            print("  " + " ".join(cmd))
            continue
        if (out_dir / "manifest.json").exists() and not args.force:
            print(f"  SKIP (manifest.json exists; use --force to rebuild): {out_dir}")
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            # Keep going so one bad variant does not hide the others' status.
            failures.append(rule)
            print(f"  FAILED (exit {result.returncode}) for rule={rule}")

    if args.dry_run:
        return

    summarize(dirs)

    if failures:
        sys.exit(f"\n{len(failures)} variant(s) FAILED: {', '.join(failures)}")
    print("\nAll variants built. Point a run at one with, e.g.:")
    print(
        f"  flwr run . --run-config \"processed-dir='{dirs[args.rules[0]]}'\""
    )


if __name__ == "__main__":
    main()
