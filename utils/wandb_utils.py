"""Centralized wandb tracking utilities for all models and experiments.

Owner: Alex Pellet

Design:
    All wandb lifecycle (init, log, finish) is managed here. Individual model
    files should never call wandb.init() or wandb.finish() directly.

    Runs are explicit objects — init_run() returns a Run that you pass into
    log_metrics(), log_params(), log_extra(), and finish_run(). This means
    multiple runs can be active simultaneously in the same process without
    interfering with each other.

    run_with_tracking() manages the full lifecycle for the common single-run
    case. For parallel or sequential multi-run workflows, use init_run() /
    finish_run() directly.

    For the federated structure, the stateless Message-API ClientApps cannot
    hold a Run across rounds, so they use log_client_round() — a resume-by-id
    helper that re-attaches to a persistent run each round, logs, and finishes.
    A federated hyperparameter sweep cannot call a single in-process train_fn
    (the loop spans ServerApp + ClientApps across processes); build_flwr_sweep_fn()
    launches `flwr run` per trial instead. See those functions below.

Typical single-model usage:
    from utils.wandb_utils import run_with_tracking

    metrics = run_with_tracking(
        train_fn=train_and_evaluate,
        train_kwargs=dict(X_train=..., y_train=..., ...),
        project="dropout-risk",
        name="xgb-centralized",
        tags=["baseline", "centralized"],
    )

Simultaneous multi-run usage:
    from utils.wandb_utils import init_run, log_metrics, finish_run

    run_a = init_run(project="dropout-risk", name="model-a")
    run_b = init_run(project="dropout-risk", name="model-b")

    metrics_a = train_model_a(...)
    log_metrics(metrics_a, run=run_a, prefix="test/")

    metrics_b = train_model_b(...)
    log_metrics(metrics_b, run=run_b, prefix="test/")

    finish_run(run_a)
    finish_run(run_b)

Parallel multi-run usage (concurrent.futures):
    from concurrent.futures import ThreadPoolExecutor
    from utils.wandb_utils import run_with_tracking

    def run_model(name, train_fn, kwargs):
        return run_with_tracking(
            train_fn=train_fn,
            train_kwargs=kwargs,
            project="dropout-risk",
            name=name,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(run_model, "model-a", train_fn_a, kwargs_a)
        fut_b = pool.submit(run_model, "model-b", train_fn_b, kwargs_b)
        metrics_a, metrics_b = fut_a.result(), fut_b.result()

Federated client per-round logging (stateless ClientApp handlers):
    from utils.wandb_utils import log_client_round, new_run_id
    # persist new_run_id() in context.state once, then each round:
    log_client_round(run_id=rid, metrics={"client_a/grad_l2": 1.23},
                     step=server_round, project="jayquery-vfl")
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Scalar metric keys safe to log directly.
# Confusion-matrix integers are handled separately as a wandb.Table.
_SCALAR_METRIC_KEYS = [
    "auc_roc",
    "accuracy",
    "f1_macro",
    "f1_binary",
    "precision",
    "recall",
    "brier_score",
    "avg_precision",
]


def _require_wandb()->Any:
    """Import and return the wandb module, raising clearly if not installed."""
    try:
        import wandb

        return wandb
    except ImportError:
        raise ImportError(
            "wandb is not installed. Install it with: pip install wandb\n"
            "Then authenticate with: wandb login"
        )


def _resolve_run(wandb:Any, run:Any)->Any:
    """Return `run` if provided, else fall back to wandb.run (the global singleton).

    Raises if neither is available, with a message that distinguishes the
    two failure modes (no run passed vs. no global run active).

    NOTE: runs created by init_run() use reinit="create_new", which does NOT
    register them as the global wandb.run. The wandb.run fallback therefore only
    works for runs created elsewhere (e.g. by wandb.agent during a sweep). Always
    pass an explicit `run` for init_run()-created runs.
    """
    if run is not None:
        return run
    if wandb.run is not None:
        return wandb.run
    raise RuntimeError(
        "No run available. Either pass an explicit `run` returned by init_run(), "
        "or ensure a global run is active via wandb.init() / init_run()."
    )


def _build_confusion_table(metrics: dict[str, Any], wandb: Any)->Any:
    """Build a labelled 2x2 wandb.Table from confusion matrix values."""
    return wandb.Table(
        columns=["", "Predicted negative", "Predicted positive"],
        data=[
            ["Actual negative", metrics.get("tn", 0), metrics.get("fp", 0)],
            ["Actual positive", metrics.get("fn", 0), metrics.get("tp", 0)],
        ],
    )


# ── public API ─────────────────────────────────────────────────────────────────


def init_run(
    project: str,
    name: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    entity: str | None = "jayquery",
    config: dict | None = None,
    group: str | None = None,
    **wandb_kwargs: dict,
)->Any:
    """Initialize a wandb run and return the Run object.

    Each call returns an independent Run object. Multiple calls can be active
    simultaneously in the same process — pass the returned object explicitly
    to log_metrics(), log_params(), log_extra(), and finish_run() to target
    the correct run.

    Args:
        project: wandb project name.
        name: Human-readable run name. If None, wandb auto-generates one.
        tags: List of tag strings for filtering runs in the wandb UI.
        notes: Freeform description logged alongside the run.
        entity: wandb team or username. Defaults to the jayquery entity.
        config: Hyperparameters / settings dict to log upfront. Can also be
                updated later via log_params().
        **wandb_kwargs: Any additional kwargs forwarded to wandb.init(). Do NOT
                pass `settings` here — this function already sets it (passing it
                again would raise a duplicate-keyword TypeError).

    Returns:
        A wandb.Run object. Pass this to all other wandb_utils functions.
    """
    wandb = _require_wandb()
    return wandb.init(
        project=project,
        name=name,
        tags=tags,
        notes=notes,
        entity=entity,
        config=config or {},
        group=group,  # ← and pass it through here
        settings=wandb.Settings(reinit="create_new"),
        **wandb_kwargs,
    )


def finish_run(run: Any | None = None) -> None:
    """Cleanly close a wandb run.

    Args:
        run: The Run object returned by init_run(). If None, closes the
             global singleton run (wandb.run). Always pass an explicit run
             when multiple runs may be active.
    """
    wandb = _require_wandb()
    resolved = _resolve_run(wandb, run)
    resolved.finish()


def log_params(
    params: dict[str, Any],
    run: Any | None = None,
) -> None:
    """Update a run's config (hyperparameters).

    Safe to call multiple times — new keys are merged, existing keys updated.

    Args:
        params: Dict of hyperparameter names → values.
        run: Target Run object. If None, uses the global singleton run.
    """
    wandb = _require_wandb()
    resolved = _resolve_run(wandb, run)
    resolved.config.update(params, allow_val_change=True)


def log_metrics(
    metrics: dict[str, Any],
    run: Any | None = None,
    prefix: str = "",
    step: int | None = None,
    log_confusion_matrix: bool = True,
) -> None:
    """Log a compute_metrics() output dict to a wandb run.

    Args:
        metrics: Output of eval_utils.compute_metrics().
        run: Target Run object. If None, uses the global singleton run.
             Always pass an explicit run when multiple runs may be active.
        prefix: String prepended to every key, e.g. "test/" → "test/auc_roc".
                Use distinct prefixes to separate val vs test metrics in the
                same run.
        step: Optional global step value. Omit for final summary metrics.
        log_confusion_matrix: If True, also logs a wandb.Table with the
                              confusion matrix.
    """
    wandb = _require_wandb()
    resolved = _resolve_run(wandb, run)

    log_dict = {f"{prefix}{k}": metrics[k] for k in _SCALAR_METRIC_KEYS if k in metrics}

    if log_confusion_matrix and all(k in metrics for k in ("tn", "fp", "fn", "tp")):
        log_dict[f"{prefix}confusion_matrix"] = _build_confusion_table(metrics, wandb)

    kwargs = {"step": step} if step is not None else {}
    resolved.log(log_dict, **kwargs)


def log_extra(
    data: dict[str, Any],
    run: Any | None = None,
    step: int | None = None,
) -> None:
    """Log arbitrary key-value pairs to a wandb run.

    Use for model-specific values that don't come from compute_metrics():
    e.g. best_iteration, learning curves, custom loss values.

    Args:
        data: Dict of key → value. Values can be scalars, strings,
              wandb.Table, wandb.Image, etc.
        run: Target Run object. If None, uses the global singleton run.
        step: Optional global step.
    """
    wandb = _require_wandb()
    resolved = _resolve_run(wandb, run)
    kwargs = {"step": step} if step is not None else {}
    resolved.log(data, **kwargs)


def run_with_tracking(
    train_fn: Callable[..., dict[str, Any]],
    train_kwargs: dict[str, Any],
    project: str,
    name: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    entity: str | None = "jayquery",
    group: str | None = None,
    hparam_key: str = "config",
    val_metrics_key: str | None = None,
    test_metrics_key: str | None = None,
    extra_log: dict[str, Any] | None = None,
    wandb_kwargs: dict | None = None,
    on_run_created: Callable[[Any, dict], None] | None = None,
) -> dict[str, Any]:
    """Wrap any train_and_evaluate function with a full wandb tracking lifecycle.

    Creates its own isolated Run object internally, so it is safe to call
    concurrently (e.g. via ThreadPoolExecutor) or back-to-back without
    interfering with other active runs.

    Args:
        train_fn: The train_and_evaluate function to call.
        train_kwargs: Keyword arguments forwarded to train_fn.
        project: wandb project name.
        name: Run name. If None, wandb auto-generates.
        tags: List of tag strings.
        notes: Freeform run description.
        entity: wandb team or username. Default is jayquery
        hparam_key: Key in train_kwargs whose value is the hyperparameter dict
                    to log as wandb config. Default "config". Set to None to
                    skip hyperparameter logging.
        val_metrics_key: If train_fn returns a dict-of-dicts, the key for the
                         validation metrics sub-dict (logged with prefix "val/").
                         If None, assumes train_fn returns a flat metrics dict.
        test_metrics_key: Same as val_metrics_key but for test metrics.
                          If both are None, the full return value is logged
                          with prefix "test/".
        extra_log: Optional dict of additional values to log after metrics.
        wandb_kwargs: Additional kwargs forwarded to wandb.init().

    Returns:
        The unmodified return value of train_fn(**train_kwargs).

    Examples:
        # Simplest case — train_fn returns a flat metrics dict
        metrics = run_with_tracking(
            train_fn=train_and_evaluate,
            train_kwargs=dict(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
                config={"max_depth": 7},
            ),
            project="baseline",
            name="xgb-centralized",
            tags=["baseline", "xgboost"],
        )

        # Two models tracked simultaneously — each gets its own run
        from concurrent.futures import ThreadPoolExecutor

        def track(name, fn, kwargs):
            return run_with_tracking(fn, kwargs, project="baseline", name=name)

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(track, "xgb",  train_xgb,  xgb_kwargs)
            f2 = pool.submit(track, "lr",   train_lr,   lr_kwargs)
            metrics_xgb = f1.result()
            metrics_lr  = f2.result()
    """
    run = init_run(
        project=project,
        name=name,
        tags=tags,
        notes=notes,
        entity=entity,
        config=train_kwargs.get(hparam_key) if hparam_key else None,
        group=group,
        **(wandb_kwargs or {}),
    )

    if on_run_created is not None:
        on_run_created(run, train_kwargs)  # caller injects run into train_kwargs here

    try:
        result = train_fn(**train_kwargs)

        if val_metrics_key is not None or test_metrics_key is not None:
            if val_metrics_key and val_metrics_key in result:
                log_metrics(result[val_metrics_key], run=run, prefix="val/")
            if test_metrics_key and test_metrics_key in result:
                log_metrics(result[test_metrics_key], run=run, prefix="test/")
        else:
            log_metrics(result, run=run, prefix="test/")

        if extra_log:
            log_extra(extra_log, run=run)

    finally:
        # Always finish the run, even if train_fn raises
        finish_run(run)

    return result


def wandb_sweep_fn(
    train_fn: Callable[..., dict[str, Any]],
    base_train_kwargs: dict[str, Any],
    project: str,
    **tracking_kwargs: dict,
) -> Callable[[], None]:
    """Return a zero-argument sweep function compatible with wandb.agent().

    NOTE: this is for the CENTRALIZED baselines, where one in-process call to
    train_fn produces the swept metric. The VFL loop spans a ServerApp and
    multiple ClientApps across processes and CANNOT be swept this way — use
    build_flwr_sweep_fn() for that.

    The returned function merges wandb.config into base_train_kwargs["config"]
    on each sweep trial.

    Args:
        train_fn: The train_and_evaluate function.
        base_train_kwargs: Base train kwargs (data arrays, fixed params).
                           The "config" key will be merged with wandb.config.
        project: wandb project name. Currently Unused.
        **tracking_kwargs: Forwarded to run_with_tracking() (name, tags, etc.)

    Returns:
        A zero-argument callable suitable for wandb.agent().

    Example:
        sweep_config = {
            "method": "bayes",
            "metric": {"name": "test/auc_roc", "goal": "maximize"},
            "parameters": {
                "max_depth":     {"values": [3, 5, 7]},
                "learning_rate": {"min": 0.01, "max": 0.2},
            },
        }
        sweep_id = wandb.sweep(sweep_config, project="dropout-risk")
        wandb.agent(
            sweep_id,
            wandb_sweep_fn(
                train_and_evaluate,
                base_train_kwargs=dict(X_train=..., y_train=..., ...),
                project="dropout-risk",
                tags=["sweep", "xgboost"],
            ),
            count=20,
        )
    """
    import wandb as _wandb

    @functools.wraps(train_fn)
    def _sweep_trial()->None:
        # wandb.agent() calls wandb.init() for us — grab that run explicitly
        run = _wandb.run
        sweep_params = dict(_wandb.config)

        merged_kwargs = dict(base_train_kwargs)
        base_config = dict(merged_kwargs.get("config") or {})
        base_config.update(sweep_params)
        merged_kwargs["config"] = base_config

        try:
            result = train_fn(**merged_kwargs)
            log_metrics(result, run=run, prefix="test/")
            log_params(base_config, run=run)
        finally:
            finish_run(run)

    return _sweep_trial


# ── federated structure: stateless-client logging + flwr sweep ──────────────────


def new_run_id() -> str:
    """Return a fresh wandb run id (16-char base-62), without starting a run.

    Persist this once in a ClientApp's context.state and reuse it every round so
    that log_client_round() always resumes the SAME run for that node.
    """
    wandb = _require_wandb()
    return wandb.util.generate_id()


def log_client_round(
    run_id: str,
    metrics: dict[str, Any],
    *,
    project: str,
    step: int | None = None,
    config: dict[str, Any] | None = None,
    name: str | None = None,
    entity: str | None = "jayquery",
    tags: list[str] | None = None,
    group: str | None = None,
    mode: str = "online",
) -> None:
    """Resume-by-id per-round logging for stateless Flower ClientApp handlers.

    A Message-API ClientApp is ephemeral — it is instantiated for one Message and
    destroyed on reply — so it cannot hold a wandb Run across rounds. This helper
    re-attaches to the run identified by `run_id` (resume="allow"), logs
    `metrics` at `step`, and finishes. Pass the SAME run_id every round (persist
    it in context.state via new_run_id()).

    Telemetry must never break training: any wandb failure here is logged and
    swallowed, and the FL round proceeds normally.

    Trade-off: this performs one init/log/finish per call. That is fine for
    research-scale simulations (tens of rounds). For very high round counts or
    many clients, prefer logging the swept metric from the ServerApp (which is
    persistent) and keep client logging to occasional diagnostics.

    Args:
        run_id: Stable run id; the same value must be used for every round.
        metrics: Flat dict of scalar (or wandb media) values to log.
        project: wandb project name.
        step: Logging step (use the FL round number). Must be monotonic.
        config: Hyperparameters to attach (static across rounds; re-sent each
                call, which is a no-op when unchanged).
        name: Run display name. If None, wandb keeps/auto-generates one.
        entity: wandb team or username.
        tags: Tag strings for filtering in the UI.
        group: Optional run group (e.g. to group both silos of one FL run).
        mode: "online", "offline", or "disabled". Use "offline"/"disabled" in CI
              or when no wandb login is available.
    """
    log_client_rounds(
        run_id,
        [[step, metrics]],
        project=project,
        config=config,
        name=name,
        entity=entity,
        tags=tags,
        group=group,
        mode=mode,
    )


def log_client_rounds(
    run_id: str,
    entries: list[list],
    *,
    project: str,
    config: dict[str, Any] | None = None,
    name: str | None = None,
    entity: str | None = "jayquery",
    tags: list[str] | None = None,
    group: str | None = None,
    mode: str = "online",
) -> None:
    """Batched variant of log_client_round: one init/log/finish for many rounds.

    Each entry is a ``[step, metrics]`` pair; steps must be monotonically
    increasing across the batch. Buffer entries client-side and flush them here
    every few rounds — the init/finish network cycle dominates per-call cost,
    so batching cuts wall-clock overhead roughly by the batch size.

    Args:
        run_id: Stable run id; the same value must be used for every flush.
        entries: List of [step, metrics] pairs, oldest first.
        (remaining args as in log_client_round)
    """
    if not entries:
        return
    wandb = _require_wandb()
    run = None
    try:
        run = wandb.init(
            project=project,
            id=run_id,
            resume="allow",
            entity=entity,
            name=name,
            tags=tags,
            group=group,
            config=config or None,
            mode=mode,
            settings=wandb.Settings(reinit="create_new"),
        )
        for step, metrics in entries:
            kwargs = {"step": int(step)} if step is not None else {}
            run.log(metrics, **kwargs)
    except Exception:  # noqa: BLE001 — telemetry must not break the FL round
        logger.warning(
            "wandb client-round logging failed (run_id=%s, %d entries); continuing.",
            run_id,
            len(entries),
            exc_info=True,
        )
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception:  # noqa: BLE001
                logger.warning("wandb run.finish() failed; continuing.", exc_info=True)


def _fmt_run_config_value(v: Any) -> str:
    """Render a Python value as a flwr --run-config TOML scalar."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "\\'") + "'"


def build_flwr_sweep_fn(
    *,
    app: str = ".",
    federation: str | None = None,
    extra_run_config: dict[str, Any] | None = None,
    pre_trial_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    meta_dir: str = "experiments/checkpoints",
    flwr_bin: str = "flwr",
) -> Callable[[], None]:
    """Zero-arg sweep-trial fn for wandb.agent(): one federated `flwr run` per trial.

    The VFL loop spans ServerApp + ClientApps across processes, so a trial can't
    call an in-process train_fn (that's wandb_sweep_fn, for the centralized
    baselines). Instead each trial submits `flwr run` with the sampled params via
    --run-config, tagged with run-tag=<trial run id> so the ServerApp writes its
    checkpoints + meta.json under meta_dir/<run id>/. The swept metric
    ("val/auc_best") and test metrics are then read from that meta.json and
    logged to the trial run. Env vars can NOT carry the run id — the ServerApp
    executes under the SuperLink, not under `flwr run`.

    pre_trial_fn(params) may pop params that are not run-config keys (e.g.
    "cutoff-week") and return extra run-config overrides (e.g. "processed-dir").
    """
    import json
    import subprocess
    from pathlib import Path

    def _trial() -> None:
        wandb = _require_wandb()
        run = wandb.run or wandb.init()  # agent-provided, else init picks up sweep params

        try:
            params: dict[str, Any] = dict(run.config)
            cfg: dict[str, Any] = dict(extra_run_config or {})
            if pre_trial_fn is not None:
                cfg.update(pre_trial_fn(params))
            cfg.update(params)
            cfg["run-tag"] = run.id
            cfg.setdefault("wandb-group", run.id)

            sweep_tag = getattr(run, "sweep_id", run.id)
            if sweep_tag:
                tag = f"sweep-{sweep_tag}"
                cfg.setdefault("wandb-sweep-tag", tag)
                if tag not in (run.tags or ()):
                    run.tags = tuple(run.tags or ()) + (tag,)

            run_config_str = " ".join(f"{k}={_fmt_run_config_value(v)}" for k, v in cfg.items())
            cmd = [flwr_bin, "run", app]
            if federation:
                cmd.append(federation)
            cmd += ["--run-config", run_config_str, "--stream"]

            logger.info("sweep trial %s -> %s", run.id, " ".join(cmd))
            # check=True so a failed trial surfaces to the agent rather than
            # being silently scored as a missing metric.
            subprocess.run(cmd, check=True)

            meta = json.loads((Path(meta_dir) / run.id / "meta.json").read_text())
            summary: dict[str, Any] = {
                f"test/{k}": v
                for k, v in (meta.get("metrics") or {}).items()
                if isinstance(v, int | float)
            }
            if meta.get("best_val_auc") is not None:
                summary["val/auc_best"] = meta["best_val_auc"]
            summary.update(meta.get("epsilon_spent") or {})
            run.log(summary)
            log_params(cfg, run=run)
        finally:
            finish_run(run)

    return _trial