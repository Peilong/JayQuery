#!/usr/bin/env python3
"""Run a wandb hyperparameter sweep over the federated VFL training loop.

Boots the deployment infra once (SuperLink + 2 SuperNodes, reusing
run_deployment's helpers), then runs a wandb agent whose trials each submit
`flwr run` with sampled params via --run-config (see build_flwr_sweep_fn).

Trials use the rate_v1 dataset in data/processed by default. "cutoff-week" is a
legacy-v0 sweep parameter (see docs/DATA_DICTIONARY.md) honored only with
--enable-cutoff-week: it re-prepares the silos into data/processed/week{K:02d}
(cached across trials) and points processed-dir there with a matching input-dim-b.

Usage:
    python scripts/run_sweep.py --count 20
    python scripts/run_sweep.py --sweep-config experiments/configs/sweep_vfl.yaml --rounds 50
    python scripts/run_sweep.py --sweep-id <id> --count 10     # resume an existing sweep
    python scripts/run_sweep.py --enable-cutoff-week           # legacy v0 cutoff sweep
"""

from __future__ import annotations

import argparse
import pickle
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from scripts.run_deployment import (
    SUPERLINK_READY_TIMEOUT_S,
    SUPERNODE_STARTUP_GRACE_S,
    register_superlink_connection,
    require_flwr_cli,
    require_ports_free,
    start_superlink,
    start_supernode,
    wait_for_control_api,
)

PROCESSED_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sweep-config", default="experiments/configs/sweep_vfl.yaml")
    p.add_argument("--sweep-id", default=None, help="Resume an existing sweep.")
    p.add_argument("--count", type=int, default=20, help="Number of trials for this agent.")
    p.add_argument("--project", default="JayQueryICC")
    p.add_argument("--entity", default="lip-etown-research")
    p.add_argument("--app", default=".")
    p.add_argument("--rounds", type=int, default=None, help="Fixed num-server-rounds per trial.")
    p.add_argument(
        "--enable-cutoff-week",
        action="store_true",
        help="Honor a sampled cutoff-week param (legacy v0 data). Default: rate_v1 only.",
    )
    p.add_argument("--connection-name", default="jayquery-local")
    p.add_argument("--base-port", type=int, default=9092)
    p.add_argument("--host", default="127.0.0.1")
    return p.parse_args()


def make_pre_trial_fn(enable_cutoff_week: bool) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the pre-trial hook: pop cutoff-week and resolve the trial's dataset."""

    def pre_trial(params: dict[str, Any]) -> dict[str, Any]:
        from utils.wandb_utils import expand_placement

        # ICC grid: placement/epsilon are sweep axes, not run-config keys.
        # Pop them first so they never reach --run-config.
        cfg: dict[str, Any] = expand_placement(params)

        week = params.pop("cutoff-week", None)
        if week is None:
            return cfg
        if not enable_cutoff_week:
            print(
                f"[sweep] WARNING: dropping sampled cutoff-week={week} — pass "
                "--enable-cutoff-week to honor it. Trial uses rate_v1 data/processed."
            )
            return cfg
        out_dir = PROCESSED_DIR / f"week{int(week):02d}"
        if not (out_dir / "silo_b.pkl").exists():
            subprocess.run(
                [sys.executable, "scripts/prepare_silos.py",
                 "--cutoff_week", str(int(week)), "--out_dir", str(out_dir)],
                check=True,
            )
        # Legacy v0 layout is wider than the rate_v1 default — read the true width
        with open(out_dir / "silo_b.pkl", "rb") as f:
            dim_b = pickle.load(f)["X_train"].shape[1]
        cfg.update({"processed-dir": str(out_dir), "input-dim-b": int(dim_b)})
        return cfg

    return pre_trial


def main() -> None:
    args = parse_args()
    require_flwr_cli()

    import wandb
    from utils.wandb_utils import build_flwr_sweep_fn

    fleet_port, control_port = args.base_port, args.base_port + 1
    clientapp_ports = {"silo_a": args.base_port + 2, "silo_b": args.base_port + 3}
    require_ports_free(
        args.host, {"fleet_api": fleet_port, "control_api": control_port, **clientapp_ports}
    )

    procs = [start_superlink(args.host, fleet_port, control_port)]

    def cleanup(*_a: object) -> None:
        print("\n[sweep] Shutting down deployment infra...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        wait_for_control_api(args.host, control_port, SUPERLINK_READY_TIMEOUT_S)
        register_superlink_connection(args.connection_name, f"{args.host}:{control_port}")
        fleet_addr = f"{args.host}:{fleet_port}"
        for pid, silo in enumerate(("silo_a", "silo_b")):
            procs.append(
                start_supernode(args.host, fleet_addr, clientapp_ports[silo], partition_id=pid)
            )
        time.sleep(SUPERNODE_STARTUP_GRACE_S)

        sweep_id = args.sweep_id
        if sweep_id is None:
            with open(args.sweep_config) as f:
                sweep_cfg = yaml.safe_load(f)
            # Fail fast instead of silently dropping the param on every trial
            if "cutoff-week" in sweep_cfg.get("parameters", {}) and not args.enable_cutoff_week:
                sys.exit(
                    "[sweep] cutoff-week is in the sweep config but --enable-cutoff-week "
                    "is not set. Remove the param for a rate_v1 sweep, or pass the flag "
                    "for a legacy v0 cutoff sweep (docs/DATA_DICTIONARY.md)."
                )
            sweep_id = wandb.sweep(sweep_cfg, project=args.project, entity=args.entity)
            print(f"[sweep] Created sweep {sweep_id}")

        extra = {"num-server-rounds": args.rounds} if args.rounds is not None else None
        wandb.agent(
            sweep_id,
            function=build_flwr_sweep_fn(
                app=args.app,
                federation=args.connection_name,
                extra_run_config=extra,
                pre_trial_fn=make_pre_trial_fn(args.enable_cutoff_week),
            ),
            count=args.count,
            project=args.project,
            entity=args.entity,
        )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
