"""PSI benchmark tests

Created by: Alex Pellet
Owned by: Wyatt Wilfong

Measures wall-clock time and bytes transmitted for the full PSI protocol
across a range of ID set sizes, matching the scalability analysis in the
research plan (Section 6, Experiment 6).

Results are printed as a table and saved to experiments/psi_timing.csv.

─────────────────────────────────────────────────────────────────────────────
Run modes:

  Full benchmark (all sizes):
      pytest psi/benchmark.py -v -s

  Single specific size (e.g. 3000 IDs):
      PSI_BENCH_N=3000 pytest psi/benchmark.py -v -s

  Direct Python execution (also supports PSI_BENCH_N):
      python psi/benchmark.py
      PSI_BENCH_N=3000 python psi/benchmark.py
─────────────────────────────────────────────────────────────────────────────
"""

import csv
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from psi.psi_client import run_psi
from psi.psi_server import run_server

# ── Configuration ─────────────────────────────────────────────────────────────

# Default sizes from research plan Section 6 Experiment 6.
# Override any single size by setting PSI_BENCH_N in the environment.
DEFAULT_SIZES = [500, 1_000, 2_000, 5_000, 10_000]

BASE_PORT = 59880  # ports BASE_PORT … BASE_PORT+len(sizes)-1 are used

OUTPUT_PATH = Path("experiments/psi_timing.csv")
CSV_COLUMNS = ["n_ids", "overlap", "time_ms", "bytes_sent"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_id_sets(n: int) -> tuple[list[str], list[str], int]:
    """Return (client_ids, server_ids, true_intersection_size) for a given n.

    Uses a 50% overlap: client owns IDs 0…n-1, server owns IDs n//2…3n//2-1.
    Overlap is IDs n//2…n-1, so intersection size = n - n//2.
    """
    client_ids = [str(i) for i in range(n)]
    server_ids  = [str(i) for i in range(n // 2, n + n // 2)]
    overlap     = n - n // 2
    return client_ids, server_ids, overlap


def _wire_bytes(n: int) -> int:
    """Total bytes transmitted for n IDs.

    Three framing messages (each = 4B count + n*32B blinded values):
      1. Client → Server: blinded client IDs
      2. Server → Client: blinded server IDs
      3. Server → Client: doubly-blinded client IDs
    """
    return 3 * (4 + n * 32)


def _run_one(n: int, port: int) -> dict:
    """Run the full PSI protocol for n IDs and return timing + metadata."""
    client_ids, server_ids, overlap = _build_id_sets(n)

    server_proc = multiprocessing.Process(
        target=run_server,
        args=(server_ids, "localhost", port),
    )
    server_proc.start()
    time.sleep(0.3)  # give server time to bind

    try:
        t0 = time.perf_counter()
        run_psi(client_ids, server_host="localhost", server_port=port)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    finally:
        server_proc.terminate()
        server_proc.join()

    return {
        "n_ids":      n,
        "overlap":    overlap,
        "time_ms":    round(elapsed_ms, 2),
        "bytes_sent": _wire_bytes(n),
    }


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'IDs':>8}  {'Overlap':>8}  {'Time (ms)':>12}  {'Bytes sent':>12}")
    print("─" * 46)
    for r in rows:
        print(
            f"{r['n_ids']:>8,}  "
            f"{r['overlap']:>8,}  "
            f"{r['time_ms']:>11.1f}ms  "
            f"{r['bytes_sent']:>10,}B"
        )


def _save_csv(rows: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[benchmark] Results saved to {OUTPUT_PATH}")


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.psi
@pytest.mark.slow
@pytest.mark.benchmark
def test_psi_benchmark() -> None:
    """Benchmark PSI protocol timing across all default ID set sizes.

    Skipped automatically when PSI_BENCH_N is set — use test_psi_benchmark_single
    instead for a targeted run.

    Run:
        pytest tests/test_psi_benchmark.py::test_psi_benchmark -v -s
    """
    if os.environ.get("PSI_BENCH_N"):
        pytest.skip("PSI_BENCH_N is set — run test_psi_benchmark_single instead")

    rows = []
    for i, n in enumerate(DEFAULT_SIZES):
        row = _run_one(n, port=BASE_PORT + i)
        rows.append(row)

    _print_table(rows)
    _save_csv(rows)


@pytest.mark.psi
@pytest.mark.slow
@pytest.mark.benchmark
def test_psi_benchmark_single() -> None:
    """Benchmark PSI protocol for a single ID count set via PSI_BENCH_N.

    Skipped automatically when PSI_BENCH_N is not set.

    Run:
        PSI_BENCH_N=3000 pytest tests/test_psi_benchmark.py::test_psi_benchmark_single -v -s
    """
    raw = os.environ.get("PSI_BENCH_N")
    if not raw:
        pytest.skip("PSI_BENCH_N not set — run test_psi_benchmark for all sizes")

    try:
        n = int(raw)
        assert n > 0
    except (ValueError, AssertionError):
        pytest.fail(f"PSI_BENCH_N must be a positive integer, got: {raw!r}")

    row = _run_one(n, port=BASE_PORT)
    _print_table([row])
    _save_csv([row])


# ── Direct execution ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw = os.environ.get("PSI_BENCH_N")

    if raw is not None:
        try:
            n = int(raw)
            assert n > 0
        except (ValueError, AssertionError):
            raise SystemExit(f"PSI_BENCH_N must be a positive integer, got: {raw!r}")
        sizes = [n]
    else:
        sizes = DEFAULT_SIZES

    rows = [_run_one(n, port=BASE_PORT + i) for i, n in enumerate(sizes)]
    _print_table(rows)
    _save_csv(rows)