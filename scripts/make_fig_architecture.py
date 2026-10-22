"""Generate Fig. 1: the two-silo split VFL architecture with every channel.

This is the figure the placement argument rests on. Its job is to show, in one
view, each message that crosses a trust boundary and which DP mechanism (if
any) protects it -- so a reader can see that the three epsilons bound three
*different* exposures rather than three views of one.

Usage:
    python scripts/make_fig_architecture.py [--out ../Paper/figs/fig_architecture.pdf]

Sized for a single IEEE column (3.5in). Matplotlib rather than TikZ to match
the other figures in figs/ and keep one toolchain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colorblind-safe; also legible in grayscale print.
C_SILO = "#dbe7f3"
C_SILO_EDGE = "#2b6cb0"
C_COORD = "#fde8d7"
C_COORD_EDGE = "#c05621"
C_DP = "#c0392b"      # anything carrying a DP mechanism
C_PLAIN = "#555555"   # unprotected channel


def _box(ax, x, y, w, h, label, sub, face, edge) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=face, edgecolor=edge, linewidth=1.1, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h - 0.052, label, ha="center", va="top",
            fontsize=7.1, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + h - 0.125, sub, ha="center", va="top",
            fontsize=5.6, color="#333333", linespacing=1.3, zorder=3)


def _arrow(ax, p0, p1, color, style="-|>", rad=0.0, lw=1.0, dashed=False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=7,
            linewidth=lw, color=color, zorder=4,
            linestyle=(0, (2.4, 1.8)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1.5, shrinkB=1.5,
        )
    )


def build(out_path: Path) -> None:
    """Draw the architecture and write it to ``out_path``."""
    fig, ax = plt.subplots(figsize=(3.45, 2.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── parties ──────────────────────────────────────────────────────────
    _box(ax, 0.000, 0.040, 0.345, 0.255, "Silo A  (registrar)",
         "18 demographic feats.\nholds the TRUE labels", C_SILO, C_SILO_EDGE)
    _box(ax, 0.655, 0.040, 0.345, 0.255, "Silo B  (LMS)",
         "26 engagement feats.\nno labels", C_SILO, C_SILO_EDGE)
    _box(ax, 0.235, 0.800, 0.530, 0.200, "Coordinator  (neutral)",
         "top model + DP-SGD  $\\varepsilon_{\\rm srv}$\nnever sees a true label",
         C_COORD, C_COORD_EDGE)

    # ── uplink: embeddings (DP-protected) ────────────────────────────────
    _arrow(ax, (0.185, 0.300), (0.330, 0.795), C_DP, rad=0.16, lw=1.25)
    ax.text(0.012, 0.660, "emb$_A$ + RR labels\nclip+Gaussian $\\varepsilon_{\\rm cut}$\n"
                          "resp. $\\varepsilon_{\\rm lab}$",
            fontsize=5.3, color=C_DP, ha="left", va="center", linespacing=1.45)

    _arrow(ax, (0.815, 0.300), (0.670, 0.795), C_DP, rad=-0.16, lw=1.25)
    ax.text(0.988, 0.660, "emb$_B$\nclip+Gaussian\n$\\varepsilon_{\\rm cut}$",
            fontsize=5.3, color=C_DP, ha="right", va="center", linespacing=1.45)

    # ── downlink: gradients (NOT protected — a stated limitation) ────────
    _arrow(ax, (0.355, 0.793), (0.235, 0.300), C_PLAIN, rad=0.16)
    ax.text(0.150, 0.430, "$\\nabla$ (no DP)", fontsize=5.2, color=C_PLAIN,
            ha="left", va="center")
    _arrow(ax, (0.645, 0.793), (0.765, 0.300), C_PLAIN, rad=-0.16)
    ax.text(0.850, 0.430, "$\\nabla$ (no DP)", fontsize=5.2, color=C_PLAIN,
            ha="right", va="center")

    # ── evaluation round-trip: only the holder can score ─────────────────
    _arrow(ax, (0.470, 0.795), (0.470, 0.690), C_PLAIN, dashed=True)
    _arrow(ax, (0.530, 0.690), (0.530, 0.795), C_PLAIN, dashed=True)
    ax.text(0.500, 0.655, "scores $\\downarrow$   metrics $\\uparrow$",
            fontsize=5.3, color=C_PLAIN, ha="center", va="top")
    ax.text(0.500, 0.598, "labels never leave Silo A",
            fontsize=5.0, color="#777777", ha="center", va="top", style="italic")

    # ── one-time PSI alignment ───────────────────────────────────────────
    _arrow(ax, (0.352, 0.150), (0.648, 0.150), C_PLAIN, style="<|-|>")
    ax.text(0.500, 0.182, "one-time PSI", fontsize=5.4, ha="center",
            va="bottom", color=C_PLAIN)
    ax.text(0.500, 0.128, "(ristretto255)", fontsize=5.0, ha="center",
            va="top", color="#777777")

    # ── legend ───────────────────────────────────────────────────────────
    ax.plot([0.055, 0.115], [0.003, 0.003], color=C_DP, lw=1.25)
    ax.text(0.126, 0.003, "DP-protected", fontsize=5.2, va="center")
    ax.plot([0.520, 0.580], [0.003, 0.003], color=C_PLAIN, lw=1.0)
    ax.text(0.591, 0.003, "unprotected", fontsize=5.2, va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        default="../Paper/figs/fig_architecture.pdf",
        help="Output PDF path (default: ../Paper/figs/fig_architecture.pdf).",
    )
    build(Path(p.parse_args().out))


if __name__ == "__main__":
    main()
