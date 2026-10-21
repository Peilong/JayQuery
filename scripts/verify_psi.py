#!/usr/bin/env python3
"""Verify the PSI protocol reproduces a variant's directly-computed intersection (§G).

Owner: Alex Pellet

WHY THIS EXISTS
---------------
`prepare_silos.py` computes the Silo A ∩ Silo B intersection **directly** and writes
`ordered_id_list.json` per variant. That decoupling was a deliberate decision
(docs/action_plan.md §E): running a multi-process cryptographic protocol inside every
dataset regeneration would make the "one command rebuilds everything" requirement
impossible, and each of the three variants has a different student set.

But the architecture *claims* PSI is what aligns the two silos without either party
revealing its ID list. If PSI never runs against the data actually used, that claim is
unverified. This script closes the gap: it runs the real ECDH protocol over a variant's
true ID sets and asserts the result matches the file the pipeline produced.

It does **not** modify `psi/` (Wyatt's module). `run_psi` / `run_server` already accept ID
lists as arguments, so the protocol is exercised exactly as shipped — only the inputs and
the output comparison are supplied here.

Usage:
    python scripts/verify_psi.py                          # dedup_first
    python scripts/verify_psi.py --variant dedup_best
    python scripts/verify_psi.py --all                    # every variant
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
DEFAULT_PORT = 50077


def silo_id_sets(variant_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Re-derive each silo's ID list for a variant, plus the expected intersection.

    Rebuilds the cohort and both silos' features from the raw CSVs using the parameters
    recorded in the variant's manifest, so the ID sets are exactly what produced the
    dataset — not an approximation.

    Args:
        variant_dir: A `data/processed/<variant>` directory containing manifest.json
            and ordered_id_list.json.

    Returns:
        Tuple of (silo_a_ids, silo_b_ids, expected_intersection), all as strings.

    Raises:
        FileNotFoundError: If the manifest or id list is missing.
    """
    from scripts.prepare_silos import (
        build_cohort,
        prepare_silo_a_data,
        prepare_silo_b_rate_v1,
    )

    manifest = json.loads((variant_dir / "manifest.json").read_text())
    modules = None if manifest["modules"] == "all" else list(manifest["modules"])

    cohort, _ = build_cohort(
        pd.read_csv(RAW / "studentInfo.csv"),
        pd.read_csv(RAW / "studentRegistration.csv"),
        modules=modules,
        dedup_rule=manifest["dedup_rule"],
        withdrawal_cutoff_day=manifest["withdrawal_cutoff_day"],
    )
    silo_a = prepare_silo_a_data(cohort)
    silo_b = prepare_silo_b_rate_v1(
        cohort,
        pd.read_csv(RAW / "studentVle.csv"),
        pd.read_csv(RAW / "vle.csv"),
        pd.read_csv(RAW / "assessments.csv"),
        pd.read_csv(RAW / "studentAssessment.csv"),
        pd.read_csv(RAW / "courses.csv"),
    )
    a_ids = [str(int(i)) for i in silo_a.index]
    b_ids = [str(int(i)) for i in silo_b["id_student"].dropna()]
    expected = [str(i) for i in json.loads((variant_dir / "ordered_id_list.json").read_text())]
    return a_ids, b_ids, expected


def run_protocol(a_ids: list[str], b_ids: list[str], port: int) -> list[str]:
    """Run the real ECDH PSI protocol in two processes and return the intersection.

    Args:
        a_ids: Silo A's student IDs (the client side).
        b_ids: Silo B's student IDs (the server side).
        port: TCP port for the PSI server.

    Returns:
        The intersection as returned by the protocol.
    """
    import tempfile

    import psi.psi_client as psi_client
    from psi.psi_client import run_psi
    from psi.psi_server import run_server

    # run_psi() writes its result to a module-level OUTPUT_PATH pointing at the REAL
    # data/processed/ordered_id_list.json. Verifying must never overwrite the artifact
    # being verified — this is the same trap that let `pytest` destroy the real PSI
    # output (docs/action_plan.md §I). Redirect before calling.
    original_output = psi_client.OUTPUT_PATH
    with tempfile.TemporaryDirectory() as tmp:
        psi_client.OUTPUT_PATH = Path(tmp) / "ordered_id_list.json"
        server = multiprocessing.Process(
            target=run_server, args=(b_ids, "localhost", port), daemon=True
        )
        server.start()
        time.sleep(2)  # let the socket bind before the client connects
        try:
            return run_psi(a_ids, server_host="localhost", server_port=port)
        finally:
            server.terminate()
            server.join(timeout=5)
            psi_client.OUTPUT_PATH = original_output


def verify(variant: str, port: int) -> bool:
    """Verify one variant. Returns True if the protocol matches the pipeline exactly."""
    variant_dir = PROCESSED / variant
    if not (variant_dir / "manifest.json").exists():
        print(f"  SKIP {variant}: no manifest.json (run scripts/prepare_variants.py)")
        return False

    print(f"\n=== {variant} ===")
    a_ids, b_ids, expected = silo_id_sets(variant_dir)
    print(f"  Silo A ids: {len(a_ids):,}   Silo B ids: {len(b_ids):,}")
    print(f"  pipeline-computed intersection: {len(expected):,}")

    t0 = time.monotonic()
    actual = run_protocol(a_ids, b_ids, port)
    print(f"  PSI protocol intersection     : {len(actual):,}  ({time.monotonic() - t0:.1f}s)")

    same_set = set(actual) == set(expected)
    same_order = list(actual) == list(expected)
    print(f"  same SET   : {same_set}   <- the security-relevant property")
    print(f"  same ORDER : {same_order}")
    if not same_set:
        only_psi = sorted(set(actual) - set(expected))[:5]
        only_pipe = sorted(set(expected) - set(actual))[:5]
        print(f"    only in PSI     ({len(set(actual) - set(expected))}): {only_psi}")
        print(f"    only in pipeline({len(set(expected) - set(actual))}): {only_pipe}")
    elif not same_order:
        print(
            "    NOTE (expected): ordering convention differs. `prepare_silos.py` sorts the\n"
            "    intersection NUMERICALLY (ints); `psi_client.run_psi` sorts LEXICOGRAPHICALLY\n"
            "    (strings). Same students, different row order. Both are internally consistent\n"
            "    because BOTH silos read the same file — but the two lists are NOT\n"
            "    interchangeable: swapping one for the other changes which rows land in\n"
            "    train/val/test. Regenerate both silos from whichever list you adopt."
        )
    # Set equality is what PSI must prove: that the correct intersection is computable
    # without either party revealing its ID list. Ordering is an internal convention.
    ok = same_set
    print(f"  RESULT: {'MATCH — PSI reproduces the pipeline intersection' if ok else 'MISMATCH'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", default="dedup_first")
    parser.add_argument("--all", action="store_true", help="Verify every dedup variant.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    if not Path("pyproject.toml").exists():
        sys.exit("Run from the repo root.")

    variants = ["dedup_first", "dedup_best", "dedup_worst"] if args.all else [args.variant]
    results = {v: verify(v, args.port + i) for i, v in enumerate(variants)}

    print("\n" + "=" * 60)
    for v, ok in results.items():
        print(f"  {v:<14} {'MATCH' if ok else 'MISMATCH/SKIP'}")
    if not all(results.values()):
        sys.exit("PSI verification FAILED — the protocol does not reproduce the pipeline output.")
    print("  PSI verification passed.")


if __name__ == "__main__":
    main()
