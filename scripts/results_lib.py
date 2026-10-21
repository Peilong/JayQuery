"""Shared loading, grouping and statistics for the results/ publish pipeline.

Owner: Alex Pellet

One place that knows how to read the experiment artifacts on disk and turn them into
grouped, comparable numbers. `analyze.py`, `figures.py` and `make_report.py` all build on
this so a number cannot disagree between the claims file, the figure and the table.

Nothing here runs an experiment — it only reads artifacts that already exist.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from scipy import stats

EXP = Path("experiments")
PROCESSED = Path("data/processed")
RESULTS = Path("results")

# A "family" is one experiment: which artifacts hold it, how to parse a run tag into a
# group, and which group is the reference every other group is compared against.
#
#   stems     result JSONs to merge, in order (a later file supersedes an earlier tag)
#   pattern   regex over the run tag; named groups except `seed` form the group key
#   split_by  key that partitions comparisons (e.g. compare arms *within* each variant)
#   arm_key   key whose values are the things being compared
#   baseline  the value of arm_key that acts as the reference
@dataclass(frozen=True)
class Family:
    """One experiment family and how to compare its arms."""

    name: str
    title: str
    stems: tuple[str, ...]
    pattern: str
    arm_key: str
    baseline: str
    split_by: str | None = None
    baseline_label: str = "ceiling"


FAMILIES: tuple[Family, ...] = (
    Family(
        name="utility_matrix",
        title="Utility matrix — DP mode x dedup variant",
        stems=("d3_dedup_first", "d3_sensitivity", "d3_matrix", "d3_rerun"),
        pattern=r"^d3_(?P<variant>[a-z_]+)_(?P<arm>no_dp|cut_only|server_only|all_dp)_s(?P<seed>\d+)$",
        arm_key="arm",
        baseline="no_dp",
        split_by="variant",
        baseline_label="no-DP ceiling",
    ),
    Family(
        name="deployment_headline",
        title="Headline arms in REAL federation mode (deployment)",
        stems=("deploy_headline",),
        pattern=r"^d3dep_(?P<variant>[a-z_]+)_(?P<arm>no_dp|cut_only|server_only|all_dp)_s(?P<seed>\d+)$",
        arm_key="arm",
        baseline="no_dp",
        split_by="variant",
        baseline_label="no-DP ceiling",
    ),
    Family(
        name="epsilon_sweep",
        title="Privacy-utility curve — cut-layer epsilon sweep",
        stems=("eps_sweep",),
        pattern=r"^eps_(?P<epsilon>[\d.]+|inf)_s(?P<seed>\d+)$",
        arm_key="epsilon",
        baseline="inf",
        baseline_label="no-noise ceiling",
    ),
    Family(
        name="clip_calibration",
        title="Clip calibration — utility at IDENTICAL epsilon",
        stems=("clip_calibration",),
        pattern=r"^clip_(?P<clip>[^_]+)_s(?P<seed>\d+)$",
        arm_key="clip",
        baseline="S1.0",
        baseline_label="S=1.0",
    ),
    Family(
        name="equivalence",
        title="Runtime equivalence — simulation vs deployment",
        stems=("l5_results",),
        pattern=r"^l5_(?P<mode>sim|deployment)_s(?P<seed>\d+)$",
        arm_key="mode",
        baseline="deployment",
        baseline_label="deployment",
    ),
)

# Metrics carried through every table, figure and claim: (meta.json path, label, decimals).
METRICS: tuple[tuple[str, str, int], ...] = (
    ("metrics.auc_roc", "AUC", 4),
    ("metrics.avg_precision", "AP", 4),
    ("metrics.recall_at_precision_70", "r@p70", 3),
    ("metrics.mcc", "MCC", 3),
    ("metrics.pred_std", "pred_std", 4),
)
PRIMARY_METRIC = "metrics.auc_roc"


@dataclass
class Group:
    """One arm of one experiment: its runs, its metric values, and its health flags."""

    key: tuple[str, ...]
    key_names: list[str]
    metas: list[dict] = field(default_factory=list)

    def label(self, name: str) -> str:
        """Value of one key component.

        Args:
            name: A key name (e.g. "arm", "variant").

        Returns:
            That component's value, or "" if this group has no such key.
        """
        return self.key[self.key_names.index(name)] if name in self.key_names else ""

    def values(self, dotted: str) -> list[float]:
        """Every run's value for one metric.

        Args:
            dotted: Dotted meta.json path (e.g. "metrics.auc_roc").

        Returns:
            The values present, skipping runs that lack the metric.
        """
        return [v for m in self.metas if (v := dig(m, dotted)) is not None]

    @property
    def n(self) -> int:
        """Number of runs in this group."""
        return len(self.metas)

    # meta.json stop_reason values that mean "a privacy mechanism ended this run", as
    # opposed to "val AUC stopped improving". Only the former makes a point under-trained
    # in a way that confounds a privacy-utility reading.
    BUDGET_REASONS = ("silo_budget", "server_budget", "combined_cap")

    @property
    def n_early(self) -> int:
        """Runs that ended before the horizon, for any reason."""
        return sum(1 for m in self.metas if m.get("stopped_early"))

    @property
    def n_budget_stopped(self) -> int:
        """Runs a PRIVACY BUDGET ended — genuinely under-trained, not merely converged.

        Runs predating stop_reason recording report None; they are counted as unknown
        rather than silently assumed to be budget stops, which is what the earlier
        reporting did (and got wrong: the Smed/Sp75 clip arms stopped at round 10-35 with
        composed epsilon 1.05 against a 3.0 cap — ordinary early stopping, not a budget).
        """
        return sum(
            1 for m in self.metas
            if m.get("stopped_early") and m.get("stop_reason") in self.BUDGET_REASONS
        )

    @property
    def n_val_stopped(self) -> int:
        """Runs ordinary validation early stopping ended — converged, not privacy-limited."""
        return sum(
            1 for m in self.metas
            if m.get("stopped_early") and m.get("stop_reason") == "val_early_stop"
        )

    @property
    def n_stop_unknown(self) -> int:
        """Runs that stopped early but predate stop_reason recording."""
        return sum(
            1 for m in self.metas if m.get("stopped_early") and not m.get("stop_reason")
        )

    @property
    def n_truncated(self) -> int:
        """Runs still improving when the horizon ended (§K5)."""
        return sum(
            1 for m in self.metas
            if int(m.get("best_round", 0)) >= int(m.get("num_rounds", 1))
            and not m.get("stopped_early")
        )

    @property
    def n_unauditable(self) -> int:
        """Runs whose meta.json carries no mechanism block, so their epsilon is unverifiable."""
        return sum(1 for m in self.metas if "mechanism" not in m)

    def caveats(self) -> list[str]:
        """Human-readable warnings that qualify this group's numbers.

        Returns:
            Zero or more short caveat strings, each naming how many runs it affects.
        """
        out = []
        if self.n_budget_stopped:
            out.append(f"{self.n_budget_stopped}/{self.n} ended by the PRIVACY BUDGET "
                       "(under-trained, not over-noised)")
        if self.n_val_stopped:
            out.append(f"{self.n_val_stopped}/{self.n} ended by validation early stopping "
                       "(converged/diverged — NOT a privacy effect)")
        if self.n_stop_unknown:
            out.append(f"{self.n_stop_unknown}/{self.n} stopped early for an UNRECORDED "
                       "reason (run predates stop_reason; re-run to disambiguate)")
        if self.n_truncated:
            out.append(f"{self.n_truncated}/{self.n} still improving at the horizon")
        if self.n_unauditable:
            out.append(f"{self.n_unauditable}/{self.n} have no mechanism record "
                       "(epsilon not verifiable from the artifact)")
        return out


def dig(meta: dict, dotted: str) -> float | None:
    """Read a dotted path out of a nested dict.

    Args:
        meta: Parsed meta.json.
        dotted: Path like "metrics.auc_roc".

    Returns:
        The numeric value, or None if any segment is missing or non-numeric.
    """
    cur: object = meta
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, int | float) else None


def sortable(v: str) -> tuple:
    """Sort key ordering numeric-looking strings numerically, with "inf" last.

    Args:
        v: A group key component.

    Returns:
        A tuple usable as a sort key.
    """
    if v == "inf":
        return (2, 0.0, "")
    try:
        return (1, float(v), "")
    except ValueError:
        return (0, 0.0, v)


def load_family(family: Family) -> dict[tuple[str, ...], Group]:
    """Load and group every run belonging to one experiment family.

    Args:
        family: The family definition.

    Returns:
        Mapping of group key -> Group, in sorted key order. Empty if no artifact exists.
    """
    runs: dict[str, dict] = {}
    for stem in family.stems:
        path = EXP / f"{stem}.json"
        if path.exists():
            runs.update(json.loads(path.read_text()))
    rx = re.compile(family.pattern)
    key_names = [g for g in rx.groupindex if g != "seed"]
    groups: dict[tuple[str, ...], Group] = {}
    for tag, meta in runs.items():
        m = rx.match(tag)
        if not m or meta is None:
            continue
        key = tuple(m.group(g) for g in key_names)
        groups.setdefault(key, Group(key=key, key_names=key_names)).metas.append(meta)
    return {k: groups[k] for k in sorted(groups, key=lambda k: [sortable(v) for v in k])}


def family_sources(family: Family) -> list[Path]:
    """Artifact paths that actually exist for a family.

    Args:
        family: The family definition.

    Returns:
        Existing result JSON paths, in merge order.
    """
    return [p for s in family.stems if (p := EXP / f"{s}.json").exists()]


def welch(arm: list[float], base: list[float]) -> dict | None:
    """Welch's unequal-variance comparison of two arms.

    Welch rather than Student because DP arms have up to 10x the seed variance of the
    no-DP ceiling, so a pooled-variance test would understate the interval.

    Args:
        arm: The arm's per-seed values.
        base: The baseline's per-seed values.

    Returns:
        Dict with delta, se, df, ci_low, ci_high, p_value and significance, or None when
        either side has fewer than two observations (no variance estimate).
    """
    if len(arm) < 2 or len(base) < 2:
        return None
    m_a, m_b = statistics.mean(arm), statistics.mean(base)
    v_a, v_b = statistics.variance(arm), statistics.variance(base)
    n_a, n_b = len(arm), len(base)
    se = math.sqrt(v_a / n_a + v_b / n_b)
    if se == 0:
        return None
    df_num = (v_a / n_a + v_b / n_b) ** 2
    df_den = (v_a / n_a) ** 2 / (n_a - 1) + (v_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den else float(n_a + n_b - 2)
    tcrit = stats.t.ppf(0.975, df)
    delta = m_a - m_b
    p = float(stats.ttest_ind(arm, base, equal_var=False).pvalue)
    return {
        "mean_arm": m_a,
        "mean_baseline": m_b,
        "delta": delta,
        "se": se,
        "df": df,
        "ci_low": delta - tcrit * se,
        "ci_high": delta + tcrit * se,
        "p_value": p,
        "significant": bool(delta + tcrit * se < 0 or delta - tcrit * se > 0),
    }


def load_audit() -> dict:
    """The privacy audit results, if verify_privacy.py has been run.

    Returns:
        Parsed experiments/privacy_audit.json, or an empty dict.
    """
    path = EXP / "privacy_audit.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_baselines() -> dict:
    """Centralized / isolated-silo baselines, merged across every seed that was run.

    Each seed writes its own ``experiments/baselines*.json``; they are merged here so the
    federation-cost comparison can carry a spread instead of resting on n=1. That number is
    the paper's headline utility claim under the feasibility framing, so a single seed is
    not enough (docs/action_plan.md section R6).

    Returns:
        ``{"results": {model: {data, metrics: {name: [values across seeds]}}}, "seeds": [...],
        "processed_dir": str}``, or an empty dict when no baselines have been run.
    """
    paths = sorted(EXP.glob("baselines*.json"))
    if not paths:
        return {}
    merged: dict[str, dict] = {}
    seeds: list[int] = []
    processed_dir = ""
    for path in paths:
        blob = json.loads(path.read_text())
        processed_dir = blob.get("processed_dir", processed_dir)
        if (seed := blob.get("seed")) is not None and seed not in seeds:
            seeds.append(int(seed))
        for name, entry in (blob.get("results") or {}).items():
            slot = merged.setdefault(name, {"data": entry.get("data", "-"), "metrics": {}})
            for metric, value in (entry.get("metrics") or {}).items():
                slot["metrics"].setdefault(metric, []).append(float(value))
    return {"results": merged, "seeds": sorted(seeds), "processed_dir": processed_dir}


def load_attack_rows() -> list[dict]:
    """Attack sweep rows, if run_attack.py --sweep has been run.

    Returns:
        One dict per attacked checkpoint, or an empty list.
    """
    path = EXP / "attack_sweep.csv"
    if not path.exists():
        return []
    import pandas as pd

    return pd.read_csv(path).to_dict(orient="records")


def group_attack_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """Group attack-sweep rows by DP arm, pooling across seeds.

    Sweep rows are one per attacked checkpoint, tagged
    ``d3_<variant>_<arm>_s<seed>``. Grouping by arm is what turns single-shot attack
    numbers into means with a spread, so an attack result can be reported the same way as
    every utility result rather than as one unreplicated observation.

    Args:
        rows: Rows from load_attack_rows().

    Returns:
        {arm: [row, ...]}, ordered no_dp, server_only, cut_only, all_dp where present.
        Rows whose tag does not parse are grouped under their raw tag.
    """
    order = ["no_dp", "server_only", "cut_only", "all_dp"]
    arm_rx = re.compile(
        r"^d3(?:dep)?_[a-z_]+?_(?P<arm>no_dp|server_only|cut_only|all_dp)_s\d+$"
    )
    # Epsilon-grid checkpoints (eps_<value>_s<seed>) group by their epsilon point. Without
    # this they fell through to the raw-tag branch, so every seed became its own group of
    # one — no pooling and no error bars on the epsilon curve.
    eps_rx = re.compile(r"^eps_(?P<eps>[\d.]+|inf)_s\d+$")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("error"):
            continue
        tag = str(row.get("eps_tag", ""))
        if m := arm_rx.match(tag):
            key = m.group("arm")
        elif m := eps_rx.match(tag):
            key = f"eps={m.group('eps')}"
        else:
            key = tag
        grouped.setdefault(key, []).append(row)

    def sort_key(name: str) -> tuple:
        """Arms first in mechanism order, then epsilon points ascending, 'inf' last."""
        if name in order:
            return (0, order.index(name), 0.0, "")
        if name.startswith("eps="):
            v = name.removeprefix("eps=")
            return (1, 0, float("inf") if v == "inf" else float(v), "")
        return (2, 0, 0.0, name)

    return {k: grouped[k] for k in sorted(grouped, key=sort_key)}


def attack_values(rows: list[dict], key: str) -> list[float]:
    """Numeric values for one attack metric across a group's rows.

    Args:
        rows: Rows for one arm.
        key: Column name (e.g. "inv_asr", "attr_gender_lift").

    Returns:
        The present numeric values, skipping blanks and NaNs.
    """
    out = []
    for row in rows:
        v = row.get(key)
        if isinstance(v, int | float) and v == v:
            out.append(float(v))
    return out


def load_variants() -> list[dict]:
    """Dataset provenance, one record per prepared variant.

    Returns:
        Each variant's manifest with its directory name added as "variant".
    """
    out = []
    for manifest in sorted(PROCESSED.glob("*/manifest.json")):
        m = json.loads(manifest.read_text())
        m["variant"] = manifest.parent.name
        out.append(m)
    return out


def fmt_mean_sd(values: list[float], decimals: int) -> str:
    """Format values as ``mean ± sd``, or the bare value when n == 1.

    Args:
        values: The observations.
        decimals: Digits after the point.

    Returns:
        A display string, "-" when empty.
    """
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{decimals}f}"
    return f"{statistics.mean(values):.{decimals}f} ± {statistics.stdev(values):.{decimals}f}"
