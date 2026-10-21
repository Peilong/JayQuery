#!/usr/bin/env python3
"""Hardened batch runner for multi-run experiments (§L5 equivalence, §D3 matrix, §M full-check).

Owner: Alex Pellet

One runner for every batch experiment. They share identical machinery — the only thing
that differs is the job list — so folding them avoids maintaining the same fragile
process handling three times.

WHY THIS EXISTS: naive batch loops silently lose runs. All four failure modes below were
observed and root-caused on 2026-07-28 (docs/action_plan.md §O); this script encodes the
fixes so they cannot be forgotten.

  O-A  `flwr-simulation` workers LEAK — one per run, ~800 MB each, never reaped. After
       ~15-18 consecutive simulation runs a 15 GB machine OOMs and runs die with
       `ray.exceptions.OutOfMemoryError`, reported as a missing meta.json rather than an
       error. => reap after every run.
  O-B  `flwr run --stream` RETURNS BEFORE `meta.json` is written. Reaping immediately kills
       the worker mid-write. => wait for the artifact, then reap.
  O-C  `pkill -f flwr-simulation` matches the FULL COMMAND LINE, so a script mentioning the
       process name matches ITSELF and kills the runner. => `pkill -x` (exact name) only.
  O-D  Redirecting run output to /dev/null destroys the evidence needed to diagnose a
       failure. => every run gets its own log file.

Usage:
    python scripts/run_matrix.py --jobs l5            # sim-vs-deployment equivalence
    python scripts/run_matrix.py --jobs d3            # full seed x variant x dp-mode matrix
    python scripts/run_matrix.py --jobs d3 --dry-run  # print the plan, run nothing
    python scripts/run_matrix.py --jobs l5 --resume   # skip jobs that already have meta.json
    python scripts/run_matrix.py --jobs d3 --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

CKPT_DIR = Path("experiments/checkpoints")
LOG_DIR = Path("experiments/matrix_logs")
PROCESSED = "data/processed"

# §K3: settled shared horizon. All configs are within the noise bar of their own optimum here.
DEFAULT_ROUNDS = 200

DP_MODES: dict[str, tuple[bool, bool]] = {
    # name          (server-dp-enabled, cut-noise-enabled)
    "no_dp": (False, False),
    "cut_only": (False, True),
    "server_only": (True, False),
    "all_dp": (True, True),
}

VARIANTS = ("dedup_first", "dedup_best", "dedup_worst")


@dataclass
class Job:
    """One run: a tag (also its checkpoint dir), a run mode, and run-config overrides."""

    tag: str
    mode: str  # "sim" | "deployment"
    config: dict[str, object] = field(default_factory=dict)
    rounds: int = DEFAULT_ROUNDS

    @property
    def meta_path(self) -> Path:
        return CKPT_DIR / self.tag / "meta.json"


def _fmt_value(v: object) -> str:
    """Render a value as a flwr --run-config scalar (strings need inner quotes)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    return "'" + str(v).replace("'", "\\'") + "'"


def build_run_config(job: Job) -> str:
    """Flatten a job's overrides into one space-separated --run-config string."""
    cfg: dict[str, object] = {
        "num-server-rounds": job.rounds,
        "run-tag": job.tag,
        **job.config,
    }
    return " ".join(f"{k}={_fmt_value(v)}" for k, v in cfg.items())


def build_command(job: Job) -> list[str]:
    """Build the argv for one job, dispatching on run mode."""
    run_config = build_run_config(job)
    if job.mode == "sim":
        return [
            "flwr", "run", ".", "local",
            "--federation-config", "num-supernodes=2",
            "--run-config", run_config,
            "--stream",
        ]
    if job.mode == "deployment":
        # num-server-rounds already lives in run_config; passing --rounds too would put
        # the same key in the string twice.
        return [
            sys.executable, "scripts/run_deployment.py",
            "--run-config", run_config,
        ]
    raise ValueError(f"unknown mode {job.mode!r} (expected 'sim' or 'deployment')")


def mem_available_gb() -> float:
    """Available system memory in GB (O-A: an approaching OOM should be visible)."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def reap_simulation_workers() -> int:
    """Kill leaked `flwr-simulation` workers. Returns how many were live beforehand.

    O-C: MUST use `pkill -x` (exact process name). `pkill -f` matches the full command
    line, so this script — which names the process in its own source — would match and
    kill itself.
    """
    live = subprocess.run(
        ["pgrep", "-x", "flwr-simulation"], capture_output=True, text=True, check=False
    )
    n = len([p for p in live.stdout.split() if p.strip()])
    if n:
        subprocess.run(["pkill", "-x", "flwr-simulation"], check=False)
        time.sleep(2)
    return n


def wait_for_meta(job: Job, timeout_s: int = 120) -> bool:
    """Wait for the run's meta.json to appear (O-B: it lands after the command returns)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if job.meta_path.exists():
            return True
        time.sleep(2)
    return job.meta_path.exists()


def run_job(job: Job, attempts: int = 2, verbose: bool = True) -> dict | None:
    """Execute one job with the full hardened lifecycle.

    Returns the parsed meta.json, or None if every attempt failed.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        if job.meta_path.parent.exists():
            subprocess.run(["rm", "-rf", str(job.meta_path.parent)], check=False)
        log_path = LOG_DIR / f"{job.tag}.attempt{attempt}.log"
        mem_before = mem_available_gb()
        if verbose:
            print(
                f"  [{job.tag}] mode={job.mode} attempt {attempt}/{attempts} "
                f"mem_avail={mem_before:.1f}GB -> {log_path}"
            )
        t0 = time.monotonic()
        # O-D: never /dev/null — a failed run's output is the only diagnostic.
        with open(log_path, "w") as fh:
            subprocess.run(build_command(job), stdout=fh, stderr=subprocess.STDOUT, check=False)
        ok = wait_for_meta(job)  # O-B
        leaked = reap_simulation_workers()  # O-A via O-C
        elapsed = time.monotonic() - t0

        if ok:
            meta = json.loads(job.meta_path.read_text())
            if verbose:
                print(
                    f"    OK  {elapsed:6.0f}s  test_auc={meta['metrics']['auc_roc']:.4f} "
                    f"best_val={meta.get('best_val_auc', float('nan')):.4f} "
                    f"(reaped {leaked}, mem_avail={mem_available_gb():.1f}GB)"
                )
            return meta
        if verbose:
            print(
                f"    FAIL {elapsed:6.0f}s — no meta.json (reaped {leaked}, "
                f"mem_avail={mem_available_gb():.1f}GB). See {log_path}"
            )
    return None


# ── job lists ────────────────────────────────────────────────────────────────


def jobs_l5(seeds: list[int], variant: str, rounds: int) -> list[Job]:
    """§L5: simulation vs deployment equivalence, paired on seed.

    `no_dp` only, deliberately: its seed variance is the smallest of any config, so it
    gives the most statistical power per (expensive) deployment run. If the two runtimes
    agree here they are computing the same thing; the DP configs then differ only by
    noise draws, which is already characterised in §K.
    """
    base: dict[str, object] = {
        "processed-dir": f"{PROCESSED}/{variant}",
        "track-wandb": False,
        "server-dp-enabled": False,
        "cut-noise-enabled": False,
    }
    return [
        Job(tag=f"l5_{mode}_s{seed}", mode=mode, config={**base, "seed": seed}, rounds=rounds)
        for mode in ("sim", "deployment")
        for seed in seeds
    ]


def jobs_clip(seeds: list[int], variant: str, rounds: int, mode: str) -> list[Job]:
    """Calibrate the cut-layer DP sensitivity S at FIXED epsilon.

    `cut-noise-max-norm` shipped at 1.0 — a documented placeholder — while measured
    embedding row norms are ~5.6 (Silo A) / ~4.0 (Silo B), so every row was clipped to a
    fifth of its length. Raising S does NOT change signal-to-noise (noise scales with S,
    so S cancels); what it changes is **clipping bias**: at S=1.0 every embedding is
    squashed to the same length, so the relative-magnitude signal between students is
    destroyed and only direction survives. This measures whether recovering that costs or
    buys utility at identical epsilon.
    """
    settings = {
        "S1.0": (1.0, 1.0),      # shipped placeholder — 100% of rows clipped
        "Smed": (5.6, 4.0),      # per-silo medians  — ~50% clipped
        "Sp75": (6.7, 6.3),      # per-silo p75      — ~25% clipped
    }
    return [
        Job(
            tag=f"clip_{name}_s{seed}",
            mode=mode,
            rounds=rounds,
            config={
                "processed-dir": f"{PROCESSED}/{variant}",
                "server-dp-enabled": False,
                "cut-noise-enabled": True,      # cut-layer only: isolate the clip's effect
                "cut-noise-target-epsilon": 1.0,
                "cut-noise-max-norm-a": sa,
                "cut-noise-max-norm-b": sb,
                "seed": seed,
                "track-wandb": False,
            },
        )
        for name, (sa, sb) in settings.items()
        for seed in seeds
    ]


def jobs_eps(seeds: list[int], variant: str, rounds: int, mode: str, group: str,
             max_norm_a: float, max_norm_b: float) -> list[Job]:
    """The privacy-utility curve: sweep cut-layer epsilon at a calibrated clip.

    Logged to wandb under `group` so the sweep is browsable. epsilon=0 is the sentinel
    for "cut noise disabled" (the no-DP ceiling of this curve).
    """
    jobs: list[Job] = []
    for eps in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        for seed in seeds:
            enabled = eps > 0
            cfg: dict[str, object] = {
                "processed-dir": f"{PROCESSED}/{variant}",
                "server-dp-enabled": False,
                "cut-noise-enabled": enabled,
                "cut-noise-max-norm-a": max_norm_a,
                "cut-noise-max-norm-b": max_norm_b,
                "seed": seed,
                "track-wandb": True,
                "wandb-mode": "offline",
                "wandb-group": group,
            }
            if enabled:
                cfg["cut-noise-target-epsilon"] = eps
            jobs.append(
                Job(tag=f"eps_{'inf' if not enabled else eps}_s{seed}",
                    mode=mode, rounds=rounds, config=cfg)
            )
    return jobs


def jobs_d3(seeds: list[int], variants: list[str], rounds: int, mode: str) -> list[Job]:
    """§D3: the reported matrix — variant x dp-mode x seed.

    Deployment jobs get a distinct `d3dep_` tag prefix. Without it a deployment run and a
    simulation run of the same cell share a checkpoint directory: the second would
    overwrite the first, and the report would merge both into one table — precisely the
    mode-mixing the project forbids (docs/FINDINGS.md section 3, rule 3).
    """
    prefix = "d3" if mode == "sim" else "d3dep"
    jobs: list[Job] = []
    for variant in variants:
        for dp_name, (server_dp, cut) in DP_MODES.items():
            for seed in seeds:
                jobs.append(
                    Job(
                        tag=f"{prefix}_{variant}_{dp_name}_s{seed}",
                        mode=mode,
                        rounds=rounds,
                        config={
                            "processed-dir": f"{PROCESSED}/{variant}",
                            "server-dp-enabled": server_dp,
                            "cut-noise-enabled": cut,
                            "seed": seed,
                            "track-wandb": True,
                            "wandb-mode": "offline",
                            "wandb-group": "d3-matrix",
                        },
                    )
                )
    return jobs


def summarize(results: dict[str, dict | None]) -> None:
    """Print a results table and flag anything that needs attention."""
    print("\n" + "=" * 78)
    print(f"{'job':<34} {'test_auc':>9} {'test_ap':>9} {'best_val':>9} {'eps':>8}")
    print("-" * 78)
    for tag, meta in results.items():
        if meta is None:
            print(f"{tag:<34} {'FAILED':>9}")
            continue
        m = meta["metrics"]
        eps = (meta.get("epsilon_spent") or {}).get("privacy/combined_epsilon")
        eps_s = f"{eps:.3f}" if eps else "-"
        bv = meta.get("best_val_auc") or float("nan")
        print(
            f"{tag:<34} {m['auc_roc']:>9.4f} {m['avg_precision']:>9.4f} {bv:>9.4f} {eps_s:>8}"
        )

    failed = [t for t, m in results.items() if m is None]
    # §K5: best_round == num_rounds with no early stop means the horizon truncated training.
    truncated = [
        t for t, m in results.items()
        if m and int(m.get("best_round", 0)) >= int(m.get("num_rounds", 1))
        and not m.get("stopped_early")
    ]
    collapsed = [t for t, m in results.items() if m and m["metrics"].get("pred_collapsed")]

    print("-" * 78)
    print(f"  completed: {sum(1 for m in results.values() if m)}/{len(results)}")
    if failed:
        print(f"  ⚠️  FAILED ({len(failed)}): {', '.join(failed)}")
    if truncated:
        print(f"  ⚠️  TRUNCATED at horizon (§K5) ({len(truncated)}): {', '.join(truncated)}")
    if collapsed:
        print(f"  ⚠️  pred_collapsed ({len(collapsed)}): {', '.join(collapsed)}")
    if not (failed or truncated or collapsed):
        print("  no failures, no truncation, no collapse.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--jobs", choices=["l5", "d3", "clip", "eps"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--variants", nargs="+", default=["dedup_first"],
        help="Dataset variants (d3). L5 uses the first only.",
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument(
        "--mode", choices=["sim", "deployment"], default="sim",
        help="Run mode for d3. L5 always runs BOTH modes by construction.",
    )
    parser.add_argument("--attempts", type=int, default=2, help="Retries per job (O2).")
    parser.add_argument("--resume", action="store_true", help="Skip jobs with a meta.json.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb-group", default="eps-sweep",
                        help="wandb group id for the epsilon sweep (--jobs eps).")
    # Defaults are 1.0 from the MEASURED calibration, not from the median. Raising S to
    # the median cost -0.243 AUC: the embedding's magnitude saturates at its natural norm
    # (~4.3) while the noise keeps scaling with S, so an oversized S is pure waste.
    parser.add_argument("--max-norm-a", type=float, default=1.0,
                        help="Silo A cut-layer clip S (default: 1.0, the calibration winner).")
    parser.add_argument("--max-norm-b", type=float, default=1.0,
                        help="Silo B cut-layer clip S (default: 1.0, the calibration winner).")
    parser.add_argument("--out", type=Path, default=Path("experiments/matrix_results.json"))
    args = parser.parse_args()

    if not Path("pyproject.toml").exists():
        sys.exit("Run from the repo root.")

    if args.jobs == "l5":
        jobs = jobs_l5(args.seeds, args.variants[0], args.rounds)
    elif args.jobs == "clip":
        jobs = jobs_clip(args.seeds, args.variants[0], args.rounds, args.mode)
    elif args.jobs == "eps":
        jobs = jobs_eps(args.seeds, args.variants[0], args.rounds, args.mode,
                        args.wandb_group, args.max_norm_a, args.max_norm_b)
    else:
        jobs = jobs_d3(args.seeds, args.variants, args.rounds, args.mode)

    print(
        f"{args.jobs}: {len(jobs)} jobs, {args.rounds} rounds, "
        f"mem_avail={mem_available_gb():.1f}GB"
    )
    if args.dry_run:
        for j in jobs:
            print(f"  {j.tag:34s} [{j.mode}]  {' '.join(build_command(j))[:120]}")
        return

    results: dict[str, dict | None] = {}
    for i, job in enumerate(jobs, 1):
        if args.resume and job.meta_path.exists():
            print(f"[{i}/{len(jobs)}] {job.tag}: SKIP (resume)")
            results[job.tag] = json.loads(job.meta_path.read_text())
            continue
        print(f"[{i}/{len(jobs)}] {job.tag}")
        results[job.tag] = run_job(job, attempts=args.attempts)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))  # checkpoint after every job

    summarize(results)
    print(f"\nresults -> {args.out}")
    if any(m is None for m in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
