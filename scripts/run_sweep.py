#!/usr/bin/env python3
"""Run a wandb hyperparameter sweep over the federated VFL training loop.

Boots the deployment infra once (SuperLink + 2 SuperNodes, reusing
run_deployment's helpers), then runs a wandb agent whose trials each submit
`flwr run` with sampled params via --run-config (see build_flwr_sweep_fn).
"cutoff-week" is not a run-config key: it re-prepares the silos into
data/processed/week{K:02d} (cached across trials) and points processed-dir there.

Usage:
    python scripts/run_sweep.py --count 20
    python scripts/run_sweep.py --sweep-config experiments/configs/sweep_vfl.yaml --rounds 50
    python scripts/run_sweep.py --sweep-id <id> --count 10     # resume an existing sweep
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
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
    p.add_argument("--project", default="jayquery-vfl")
    p.add_argument("--entity", default="jayquery")
    p.add_argument("--app", default=".")
    p.add_argument("--rounds", type=int, default=None, help="Fixed num-server-rounds per trial.")
    p.add_argument("--connection-name", default="jayquery-local")
    p.add_argument("--base-port", type=int, default=9092)
    p.add_argument("--host", default="127.0.0.1")
    return p.parse_args()


def prepare_cutoff_week(params: dict[str, Any]) -> dict[str, Any]:
    """Pop cutoff-week from the sampled params; build (cached) its processed dir."""
    week = params.pop("cutoff-week", None)
    if week is None:
        return {}
    out_dir = PROCESSED_DIR / f"week{int(week):02d}"
    if not (out_dir / "silo_b.pkl").exists():
        subprocess.run(
            [sys.executable, "scripts/prepare_silos.py",
             "--cutoff_week", str(int(week)), "--out_dir", str(out_dir)],
            check=True,
        )
    return {"processed-dir": str(out_dir)}


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
                sweep_id = wandb.sweep(yaml.safe_load(f), project=args.project, entity=args.entity)
            print(f"[sweep] Created sweep {sweep_id}")

        extra = {"num-server-rounds": args.rounds} if args.rounds is not None else None
        wandb.agent(
            sweep_id,
            function=build_flwr_sweep_fn(
                app=args.app,
                federation=args.connection_name,
                extra_run_config=extra,
                pre_trial_fn=prepare_cutoff_week,
            ),
            count=args.count,
            project=args.project,
            entity=args.entity,
        )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
