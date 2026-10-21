#!/usr/bin/env python3
"""Draw the paper's figures from the run artifacts.

Owner: Alex Pellet

WHY THIS EXISTS
---------------
The privacy-utility curve is the paper's core figure and it existed only as a 7-row markdown
table. A table of a curve is not a curve. Everything here reads the same artifacts the
claims and tables come from (scripts/results_lib.py), so a figure cannot disagree with the
number printed next to it.

Figures follow the plan in docs/privacy_presentation_plan.md section 5:

  fig1  privacy-utility curve   AUC and recall@p70 vs epsilon, truncation confounds marked
  fig2  epsilon isocline        utility spread at IDENTICAL epsilon (the strongest result)
  fig3  attack by defense       ASR per DP mode with its floor, so the y-axis is readable
  fig4  variance vs epsilon     seed sd explodes as the budget tightens
  fig5  attack vs epsilon       inversion ASR across the budget grid — the privacy companion
                                to fig1; measured to be a STEP, not a curve

Truncated and budget-exhausted points are drawn HOLLOW rather than dropped: hiding them
would overstate the curve, and dropping them would hide that the low-epsilon end is partly
a horizon effect. The legend says so.

Usage:
    python scripts/figures.py
    python scripts/figures.py --out-dir results/figures --format pdf
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display in this environment
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from results_lib import (  # noqa: E402
    FAMILIES,
    RESULTS,
    Family,
    attack_values,
    group_attack_rows,
    load_attack_rows,
    load_family,
    sortable,
)

# One accessible, print-safe palette shared by every figure.
INK = "#1b1b1f"
ACCENT = "#2f6f9f"
WARM = "#c2571a"
MUTED = "#8a8a94"
GRID = "#dcdce2"


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the shared axis styling.

    Args:
        ax: The axes to style.
        title: Axis title.
        xlabel: X label.
        ylabel: Y label.
    """
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK)


def _family(name: str) -> Family | None:
    """Look up a family definition by name.

    Args:
        name: The family name.

    Returns:
        The Family, or None if unknown.
    """
    return next((f for f in FAMILIES if f.name == name), None)


def _eps_series(metric: str) -> tuple[list[float], list[float], list[float], list[bool]]:
    """Epsilon-sweep points for one metric, with a flag for compromised points.

    The "inf" (no-noise) arm is excluded from the x-series and returned separately by
    callers that want it as a reference line.

    Args:
        metric: Dotted metric path.

    Returns:
        Tuple of (epsilons, means, sds, compromised) sorted by epsilon.
    """
    groups = load_family(_family("epsilon_sweep"))
    xs, ys, es, bad = [], [], [], []
    for key in sorted(groups, key=lambda k: [sortable(v) for v in k]):
        label = key[0]
        if label == "inf":
            continue
        vals = groups[key].values(metric)
        if not vals:
            continue
        xs.append(float(label))
        ys.append(statistics.mean(vals))
        es.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        # A point is compromised when its runs did not train the full horizon: either the
        # budget gate or validation early stopping ended them before the last round.
        bad.append(groups[key].n_early > 0)
    return xs, ys, es, bad


def _ceiling(metric: str) -> float | None:
    """The epsilon-sweep's no-noise reference value for one metric.

    Args:
        metric: Dotted metric path.

    Returns:
        Mean of the "inf" arm, or None if absent.
    """
    groups = load_family(_family("epsilon_sweep"))
    for key, group in groups.items():
        if key[0] == "inf" and (vals := group.values(metric)):
            return statistics.mean(vals)
    return None


def fig_privacy_utility(path: Path) -> bool:
    """Figure 1: AUC and recall@p70 against epsilon, with truncation marked.

    Args:
        path: Output file path.

    Returns:
        True if the figure was written.
    """
    series = [
        ("metrics.auc_roc", "AUC", ACCENT, "o"),
        ("metrics.recall_at_precision_70", "recall @ precision 0.70", WARM, "s"),
    ]
    if not _eps_series(series[0][0])[0]:
        return False

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for metric, label, colour, marker in series:
        xs, ys, es, bad = _eps_series(metric)
        ax.errorbar(xs, ys, yerr=es, color=colour, linewidth=1.6, zorder=3,
                    capsize=3, elinewidth=1.0, label=label)
        # Solid = ran the full horizon; hollow = at least one seed stopped early.
        for x, y, is_bad in zip(xs, ys, bad, strict=True):
            ax.plot(x, y, marker=marker, markersize=6.5, zorder=4, color=colour,
                    markerfacecolor="white" if is_bad else colour,
                    markeredgecolor=colour, markeredgewidth=1.4)
        if (ceil := _ceiling(metric)) is not None:
            ax.axhline(ceil, color=colour, linestyle=":", linewidth=1.2, alpha=0.8)
            # Label just under its own line at the left edge; the two ceilings sit ~0.016
            # apart, so labelling them at the same height on the right made them collide.
            ax.text(0.012, ceil - 0.028, f"no-noise {label.split()[0]} = {ceil:.3f}",
                    transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7.5, color=colour)

    ax.set_xscale("log")
    _style(ax, "Privacy-utility trade-off — cut-layer noise",
           "epsilon (cut-layer, per silo)", "test-set value")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.text(0.01, 0.015,
             "Hollow markers: at least one seed stopped before the full horizon (validation "
             "early stopping or the budget gate — see t4).\nDotted lines: the no-noise ceiling. "
             "Error bars: sd across seeds.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def fig_isocline(path: Path) -> bool:
    """Figure 2: utility spread at an IDENTICAL formal guarantee.

    The paper's sharpest privacy point: these configurations all report epsilon = 1.0, yet
    their utility differs by ~0.3 AUC. The clip S does not appear in the epsilon
    calculation at all, so that spread is free — it costs no privacy.

    Args:
        path: Output file path.

    Returns:
        True if the figure was written.
    """
    points: list[tuple[str, float, float]] = []

    clip = load_family(_family("clip_calibration"))
    for key, group in clip.items():
        if vals := group.values("metrics.auc_roc"):
            points.append((f"cut noise, clip {key[0]}", statistics.mean(vals),
                           statistics.stdev(vals) if len(vals) > 1 else 0.0))

    # The server-side mechanism at the same nominal epsilon, from the utility matrix.
    matrix = load_family(_family("utility_matrix"))
    for key, group in matrix.items():
        if group.label("variant") != "dedup_first":
            continue
        if group.label("arm") == "server_only" and (vals := group.values("metrics.auc_roc")):
            points.append(("server DP-SGD", statistics.mean(vals),
                           statistics.stdev(vals) if len(vals) > 1 else 0.0))
    if len(points) < 2:
        return False

    points.sort(key=lambda p: p[1])
    labels = [p[0] for p in points]
    values = [p[1] for p in points]
    errs = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ypos = range(len(points))
    ax.errorbar(values, list(ypos), xerr=errs, fmt="o", color=ACCENT,
                markersize=7, capsize=3, elinewidth=1.0, zorder=3)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=8.5)
    spread = max(values) - min(values)
    _style(ax, f"Same guarantee (epsilon = 1.0), {spread:.2f} AUC apart",
           "test AUC", "")
    fig.text(0.01, 0.02,
             "Every configuration above reports an IDENTICAL formal epsilon of 1.0. The clip S "
             "does not enter the\nepsilon calculation, so this spread is free — it buys utility "
             "at no privacy cost.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def fig_attack(path: Path) -> bool:
    """Figure 3: attack success against epsilon, with the uninformed-attacker floor.

    Without the floor an ASR is unreadable — 0.10 looks like "10% of students compromised"
    when the floor may be 0.15, i.e. worse than guessing.

    Args:
        path: Output file path.

    Returns:
        True if the figure was written (False when no attack sweep exists yet).
    """
    grouped = group_attack_rows(load_attack_rows())
    if not grouped:
        return False

    # Pool across the seeds whose checkpoints were attacked; error bars are the spread
    # across seeds when there is more than one, else the within-run Wilson interval.
    labels, asr, lo, hi, floor, label_acc = [], [], [], [], [], []
    for arm, rows in grouped.items():
        vals = attack_values(rows, "inv_asr")
        if not vals:
            continue
        mean = statistics.mean(vals)
        if len(vals) > 1:
            half = statistics.stdev(vals)
        else:
            half = max(
                mean - (attack_values(rows, "inv_asr_ci_low") or [mean])[0],
                (attack_values(rows, "inv_asr_ci_high") or [mean])[0] - mean,
            )
        labels.append(f"{arm}\n(n={len(vals)})")
        asr.append(mean)
        lo.append(half)
        hi.append(half)
        floor.append(statistics.mean(attack_values(rows, "inv_asr_majority") or [0.0]))
        if lab := attack_values(rows, "label_acc"):
            label_acc.append(statistics.mean(lab))

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    xs = list(range(len(labels)))
    ax.errorbar(xs, asr, yerr=[lo, hi], fmt="o", color=ACCENT, markersize=7,
                capsize=4, elinewidth=1.1, zorder=4, label="inversion ASR (95% CI)")
    ax.plot(xs, floor, linestyle="--", color=MUTED, linewidth=1.4, zorder=3,
            label="floor: always guess the modal value")
    if label_acc:
        ax.plot(xs[:len(label_acc)], label_acc, marker="^", color=WARM, linewidth=1.4,
                zorder=3, label="label inference (undefended channel)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.02)
    _style(ax, "Attack success against the transmitted embedding",
           "defense configuration", "attack success rate")
    ax.legend(fontsize=8, frameon=False, loc="center left")
    fig.text(0.01, 0.02,
             "Attacks consume the WIRE representation (clipped + noised), using each run's own "
             "recorded defense\nparameters. ASR at or below the floor means no leakage beyond an "
             "uninformed attacker.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def fig_variance(path: Path) -> bool:
    """Figure 4: seed-to-seed sd against epsilon.

    A deployment argument that does not depend on the attacks landing: at tight budgets the
    model is not merely worse, it is unpredictable.

    Args:
        path: Output file path.

    Returns:
        True if the figure was written.
    """
    xs, _, es, bad = _eps_series("metrics.auc_roc")
    if not xs:
        return False

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(xs, es, color=WARM, linewidth=1.6, zorder=3)
    for x, e, is_bad in zip(xs, es, bad, strict=True):
        ax.plot(x, e, marker="o", markersize=6.5, zorder=4, color=WARM,
                markerfacecolor="white" if is_bad else WARM,
                markeredgecolor=WARM, markeredgewidth=1.4)
    groups = load_family(_family("epsilon_sweep"))
    for key, group in groups.items():
        if key[0] == "inf" and len(vals := group.values("metrics.auc_roc")) > 1:
            ax.axhline(statistics.stdev(vals), color=MUTED, linestyle=":", linewidth=1.2)
            ax.text(ax.get_xlim()[1], statistics.stdev(vals), "  no-noise sd",
                    va="center", fontsize=7.5, color=MUTED)
    ax.set_xscale("log")
    _style(ax, "Training becomes unpredictable as the budget tightens",
           "epsilon (cut-layer, per silo)", "sd of test AUC across seeds")
    fig.text(0.01, 0.02,
             "An institution retraining on the same data at a tight budget could get materially "
             "different results.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def fig_attack_vs_epsilon(path: Path) -> bool:
    """Figure 5: inversion success against the cut-layer epsilon budget.

    The privacy companion to fig1. Plots the epsilon-grid attack points with the
    always-guess-modal floor and the no-noise reference, so the shape of the privacy
    trade-off is visible rather than inferred from a table.

    The measured shape is a STEP, not a curve: across every tested budget the injected
    noise still dominates the per-component signal (the clip caps the whole embedding at
    S over 32 dimensions, so each component carries only ~S/sqrt(32)), and the attack sees
    nothing anywhere. Only removing the noise entirely restores it. The annotation says so
    on the figure, because a reader seeing a flat line will otherwise assume a broken axis.

    Args:
        path: Output file path.

    Returns:
        True if the figure was written (False when no epsilon-grid rows exist).
    """
    grouped = group_attack_rows(load_attack_rows())
    points = {k: v for k, v in grouped.items() if k.startswith("eps=")}
    if len(points) < 2:
        return False

    xs, ys, es = [], [], []
    ceiling = None
    for name, rows in points.items():
        vals = attack_values(rows, "inv_asr")
        if not vals:
            continue
        eps = name.removeprefix("eps=")
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if eps == "inf":          # the no-noise arm is a reference line, not an x point
            ceiling = (mean, sd)
            continue
        xs.append(float(eps))
        ys.append(mean)
        es.append(sd)
    if not xs:
        return False

    floor = statistics.mean(
        [v for rows in points.values() for v in attack_values(rows, "inv_asr_majority")]
    )
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs, ys, es = [xs[i] for i in order], [ys[i] for i in order], [es[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.errorbar(xs, ys, yerr=es, color=ACCENT, marker="o", markersize=7, linewidth=1.6,
                capsize=4, elinewidth=1.1, zorder=4, label="inversion ASR (sd across seeds)")
    ax.axhline(floor, color=MUTED, linestyle="--", linewidth=1.4, zorder=3,
               label="floor: always guess the modal value")
    if ceiling is not None:
        ax.axhline(ceiling[0], color=WARM, linestyle=":", linewidth=1.6, zorder=3,
                   label=f"no cut-layer noise ({ceiling[0]:.3f})")
    ax.set_xscale("log")
    # Label the epsilon values actually swept. Matplotlib's default log formatter renders
    # them as 6x10^-1 / 2x10^0, which is unreadable for a budget grid.
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.minorticks_off()
    ax.set_ylim(0, max(ceiling[0] if ceiling else 0.5, max(ys)) * 1.25)
    _style(ax, "Inversion success vs the cut-layer privacy budget",
           "epsilon (cut-layer, per silo)", "attack success rate")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    fig.text(0.01, 0.015,
             "Flat by design, not a broken axis: the clip caps the embedding at S over 32 "
             "dimensions (~S/sqrt(32) per\ncomponent), so noise dominates the signal at every "
             "budget tested. The trade-off is a STEP — noise vs none.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


FIGURES = (
    ("fig1_privacy_utility_curve", fig_privacy_utility),
    ("fig2_epsilon_isocline", fig_isocline),
    ("fig3_attack_vs_epsilon", fig_attack),
    ("fig4_variance_vs_epsilon", fig_variance),
    ("fig5_attack_vs_epsilon", fig_attack_vs_epsilon),
)


def build_all(out_dir: Path, fmt: str = "png") -> list[str]:
    """Draw every figure whose source artifacts exist.

    Args:
        out_dir: Directory to write into (created if absent).
        fmt: File extension / matplotlib format.

    Returns:
        Names of the figures that were written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in FIGURES:
        path = out_dir / f"{name}.{fmt}"
        if fn(path):
            written.append(path.name)
            print(f"  {path}")
        else:
            print(f"  (skipped {name} — source artifact missing)")
    return written


def main() -> None:
    """CLI entry point: draw every figure into results/figures/."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=RESULTS / "figures")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    if not Path("pyproject.toml").exists():
        sys.exit("Run from the repo root.")

    written = build_all(args.out_dir, args.format)
    print(f"\n{len(written)} figure(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
