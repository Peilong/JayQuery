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

# Boxes are distinguished by FILL COLOUR; channels by line style and routing.
# The uplink and downlink are drawn on separate tracks rather than mirrored
# arcs, because opposite-curvature arcs between the same two boxes cross near
# their midpoints and the solid and dashed lines then sit on top of each other.
C_SILO = "#dbe7f3"
C_SILO_EDGE = "#2b6cb0"
C_COORD = "#fde8d7"
C_COORD_EDGE = "#c05621"
INK = "#222222"


def _box(ax, x, y, w, h, label, sub, face, edge) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor=face, edgecolor=edge, linewidth=1.1, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h * 0.72, label, ha="center", va="center",
            fontsize=6.2, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h * 0.34, sub, ha="center", va="center",
            fontsize=4.8, color="#333333", linespacing=1.3, zorder=4)


def _arrow(ax, p0, p1, style="-|>", rad=0.0, lw=1.0, dashed=False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=6.5,
            linewidth=lw, color=INK, zorder=5,
            linestyle=(0, (2.2, 1.8)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1.5, shrinkB=1.5,
        )
    )


def build(out_path: Path) -> None:
    """Draw the architecture and write it to ``out_path``."""
    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    # Limits exceed the content so rounded corners and arrowheads never clip.
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.12, 1.06)
    ax.axis("off")

    _box(ax, 0.000, 0.03, 0.335, 0.42, "Silo A (registrar)",
         "18 demographic feats.\nholds true labels", C_SILO, C_SILO_EDGE)
    _box(ax, 0.665, 0.03, 0.335, 0.42, "Silo B (LMS)",
         "26 engagement feats.\nno labels", C_SILO, C_SILO_EDGE)
    _box(ax, 0.255, 0.72, 0.490, 0.28, "Coordinator (neutral)",
         "top model, DP-SGD $\\varepsilon_{\\rm srv}$\nnever sees a true label",
         C_COORD, C_COORD_EDGE)

    # Uplink, DP-protected: straight, on the OUTER track.
    _arrow(ax, (0.120, 0.455), (0.285, 0.715), lw=1.35)
    ax.text(0.008, 0.625, "emb$_A$ + RR labels\n$\\varepsilon_{\\rm cut}$, "
                          "$\\varepsilon_{\\rm lab}$",
            fontsize=4.9, ha="left", va="center", linespacing=1.25, color=INK)
    _arrow(ax, (0.880, 0.455), (0.715, 0.715), lw=1.35)
    ax.text(0.992, 0.625, "emb$_B$\n$\\varepsilon_{\\rm cut}$",
            fontsize=4.9, ha="right", va="center", linespacing=1.25, color=INK)

    # Downlink, unprotected: straight, on the INNER track. Parallel to the
    # uplink and clearly offset, so the two never overlap.
    _arrow(ax, (0.395, 0.713), (0.250, 0.455), lw=0.85, dashed=True)
    _arrow(ax, (0.605, 0.713), (0.750, 0.455), lw=0.85, dashed=True)
    ax.text(0.500, 0.600, "$\\nabla$ down (no DP)", fontsize=5.0,
            ha="center", va="center", color=INK)

    _arrow(ax, (0.342, 0.185), (0.658, 0.185), style="<|-|>", lw=0.85)
    ax.text(0.500, 0.215, "PSI (ristretto255)", fontsize=4.9,
            ha="center", va="bottom", color=INK)

    ax.plot([0.045, 0.110], [-0.080, -0.080], color=INK, lw=1.35)
    ax.text(0.122, -0.080, "DP-protected", fontsize=4.9, va="center")
    ax.plot([0.530, 0.595], [-0.080, -0.080], color=INK, lw=0.85,
            linestyle=(0, (2.2, 1.8)))
    ax.text(0.607, -0.080, "unprotected", fontsize=4.9, va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
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
