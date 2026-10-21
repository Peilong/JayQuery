#!/usr/bin/env python3
"""Launch the JayQuery VFL deployment: one SuperLink + two SuperNodes
(Silo A, Silo B), register the SuperLink as a named connection, then submit
the run via `flwr run`.

Verified against flwr==1.28.0's actual CLI (flower-superlink --help,
flower-supernode --help, flwr run --help) — NOT the older docs flow that
used [tool.flwr.federations] in pyproject.toml. As of ~1.28, SuperLink
connections live in the central Flower CLI config (~/.flwr/config.toml,
managed by `flwr config`), independent of any project. There is no `flwr
config add` subcommand — this script appends the connection entry directly.

Usage (everything has a sensible default — only --app needs to exist):
    python run_deployment.py
    python run_deployment.py --rounds 10
    python run_deployment.py --app . --connection-name jayquery-local

What this does, in order:
    1. Pick free-ish local ports for the SuperLink's control/fleet APIs and
       each SuperNode's ClientAppIo API (override with --base-port if your
       defaults collide with something else already running).
    2. Start `flower-superlink --insecure` with those addresses.
    3. Register/refresh a [superlink.<name>] entry in ~/.flwr/config.toml
       pointing at the SuperLink's control API.
    4. Start two `flower-supernode` processes, one per silo, each with its
       own --clientappio-api-address and --node-config "partition-id=N".
    5. Wait for both SuperNodes to report ready, then run
       `flwr run <app> <name> --run-config "num-server-rounds=<rounds>"`.
    6. On Ctrl-C (or when flwr run exits), terminate the SuperLink and both
       SuperNodes.

This is local-machine, --insecure testing only. Do not use --insecure for
anything beyond development on a trusted machine.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
from contextlib import closing
from pathlib import Path

FLWR_CONFIG_PATH = Path.home() / ".flwr" / "config.toml"
SUPERLINK_READY_TIMEOUT_S = 20
# Both SuperNodes must REGISTER with the SuperLink before `flwr run` submits,
# or the ServerApp sees 1 node and aborts. 3s was not enough on a cold start
# (observed Oct 2026: both processes started, only one registered in time).
SUPERNODE_STARTUP_GRACE_S = 10  # override with --supernode-grace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--app", default=".", help="Path to the Flower app (default: current directory).")
    p.add_argument("--rounds", type=int, default=None, help="Override num-server-rounds for this run.")
    p.add_argument(
        "--run-config",
        default=None,
        help="Extra run-config overrides passed through to `flwr run`, as one "
        "space-separated string of k=v pairs. String values need inner quotes, e.g. "
        "--run-config \"processed-dir='data/processed/dedup_first' "
        "server-dp-enabled=true\". Merged with --rounds.",
    )
    p.add_argument(
        "--connection-name",
        default="jayquery-local",
        help="Name to register this SuperLink under in ~/.flwr/config.toml (default: jayquery-local).",
    )
    p.add_argument(
        "--base-port",
        type=int,
        default=9092,
        help=(
            "Starting port. Allocates base_port=fleet API, base_port+1=control API, "
            "base_port+2/+3=SuperNode ClientAppIo APIs (default: 9092, matching Flower's "
            "documented convention; only the relative offsets matter)."
        ),
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind/connect host for all components (default: 127.0.0.1).")
    p.add_argument(
        "--supernode-grace",
        type=float,
        default=SUPERNODE_STARTUP_GRACE_S,
        help="Seconds to wait for both SuperNodes to register before submitting "
        f"the run (default: {SUPERNODE_STARTUP_GRACE_S}). Raise this if the "
        "ServerApp aborts with 'Expected exactly 2 SuperNodes'.",
    )
    p.add_argument(
        "--skip-port-check",
        action="store_true",
        help="Don't verify ports are free before launching (faster, less safe).",
    )
    return p.parse_args()


def port_is_free(host: str, port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        return s.connect_ex((host, port)) != 0  # nonzero -> nothing listening -> free


def require_ports_free(host: str, ports: dict[str, int]) -> None:
    busy = {name: p for name, p in ports.items() if not port_is_free(host, p)}
    if busy:
        lines = "\n".join(f"  {name}: {host}:{p}" for name, p in busy.items())
        sys.exit(
            f"Port(s) already in use, refusing to start:\n{lines}\n"
            "Free them, or pick a different --base-port, or pass --skip-port-check."
        )


def require_flwr_cli() -> None:
    missing = [exe for exe in ("flower-superlink", "flower-supernode", "flwr") if shutil.which(exe) is None]
    if missing:
        sys.exit(
            f"Missing required executable(s) on PATH: {', '.join(missing)}. "
            "Activate the environment where `flwr` is installed and try again."
        )


def register_superlink_connection(name: str, control_api_address: str) -> None:
    """Add/refresh a [superlink.<name>] entry in ~/.flwr/config.toml.

    There is no `flwr config add` command at this version — the file is
    edited directly. `flwr config list` (run once) ensures the file exists
    with its default content before we touch it.
    """
    subprocess.run(["flwr", "config", "list"], capture_output=True, check=False)

    FLWR_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = FLWR_CONFIG_PATH.read_text() if FLWR_CONFIG_PATH.exists() else ""

    with FLWR_CONFIG_PATH.open("rb") as f:
        existing = tomllib.load(f) if FLWR_CONFIG_PATH.exists() else {}
    current_address = existing.get("superlink", {}).get(name, {}).get("address")

    if current_address == control_api_address:
        return  # already correctly registered, nothing to do

    section_header = f"[superlink.{name}]"
    new_section = f'{section_header}\naddress = "{control_api_address}"\ninsecure = true\n'

    if section_header in text:
        # Replace the existing section, locating its end (the next "[" line
        # or end of file).
        start = text.index(section_header)
        rest = text[start + len(section_header):]
        next_bracket = rest.find("\n[")
        end = start + len(section_header) + (next_bracket if next_bracket != -1 else len(rest))
        text = text[:start] + new_section.rstrip("\n") + text[end:]
    else:
        if text and not text.endswith("\n\n"):
            text += "\n" if text.endswith("\n") else "\n\n"
        text += new_section

    FLWR_CONFIG_PATH.write_text(text)
    print(f"[deploy] Registered SuperLink connection '{name}' -> {control_api_address}")


def start_superlink(host: str, fleet_port: int, control_port: int) -> subprocess.Popen:
    cmd = [
        "flower-superlink",
        "--insecure",
        "--fleet-api-address", f"{host}:{fleet_port}",
        "--control-api-address", f"{host}:{control_port}",
    ]
    print(f"[deploy] Starting SuperLink: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def start_supernode(host: str, superlink_fleet_addr: str, clientappio_port: int, partition_id: int) -> subprocess.Popen:
    cmd = [
        "flower-supernode",
        "--insecure",
        "--superlink", superlink_fleet_addr,
        "--clientappio-api-address", f"{host}:{clientappio_port}",
        "--node-config", f"partition-id={partition_id}",
    ]
    print(f"[deploy] Starting SuperNode (partition-id={partition_id}): {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def wait_for_control_api(host: str, port: int, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not port_is_free(host, port):  # something is now listening
            return
        time.sleep(0.5)
    sys.exit(f"SuperLink control API never came up on {host}:{port} after {timeout_s}s.")


def main() -> None:
    args = parse_args()
    require_flwr_cli()

    fleet_port = args.base_port
    control_port = args.base_port + 1
    clientapp_ports = {"silo_a": args.base_port + 2, "silo_b": args.base_port + 3}

    if not args.skip_port_check:
        require_ports_free(
            args.host,
            {"fleet_api": fleet_port, "control_api": control_port, **clientapp_ports},
        )

    superlink_proc = start_superlink(args.host, fleet_port, control_port)
    procs = [superlink_proc]

    def cleanup(*_a) -> None:
        print("\n[deploy] Shutting down...")
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

        supernode_a = start_supernode(args.host, f"{args.host}:{fleet_port}", clientapp_ports["silo_a"], partition_id=0)
        supernode_b = start_supernode(args.host, f"{args.host}:{fleet_port}", clientapp_ports["silo_b"], partition_id=1)
        procs += [supernode_a, supernode_b]

        # NOTE: this is a fixed grace period, not a real readiness check —
        # flower-supernode doesn't expose a simple local "ready" signal to
        # poll the way the SuperLink's control API does. If `flwr run` below
        # fails with a "no SuperNodes connected" style error, increase
        # SUPERNODE_STARTUP_GRACE_S or check each SuperNode's own log output.
        time.sleep(args.supernode_grace)

        # --run-config is merged into ONE flag: flwr accepts a single space-separated
        # "k=v k=v" string, and passing the flag twice would silently drop the first.
        overrides: list[str] = []
        if args.rounds is not None:
            overrides.append(f"num-server-rounds={args.rounds}")
        if args.run_config:
            overrides.append(args.run_config)

        run_cmd = ["flwr", "run", args.app, args.connection_name, "--stream"]
        if overrides:
            run_cmd += ["--run-config", " ".join(overrides)]

        print(f"[deploy] Submitting run: {' '.join(run_cmd)}")
        result = subprocess.run(run_cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    finally:
        cleanup()


if __name__ == "__main__":
    main()