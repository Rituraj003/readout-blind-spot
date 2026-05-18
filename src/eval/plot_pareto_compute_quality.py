"""Render the 129M fixed-depth compute-quality frontier used in paper_v6.

The values are the three-seed means reported in Section 5. Throughput is the
single-harness timing used for the variable-depth comparison.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def main() -> None:
    out_dir = _PROJECT_ROOT / "paper_v6" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    throughputs = {4: 21_000, 3: 28_000, 2: 41_000, 1: 80_000}
    curves = {
        "Per-loop + RMSNorm": {
            "ppl": {4: 4.97, 3: 4.95, 2: 4.95, 1: 4.98},
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
        },
        "Per-loop + raw": {
            "ppl": {4: 4.55, 3: 4.54, 2: 4.55, 1: 4.68},
            "color": "#2ca02c",
            "marker": "s",
            "linestyle": "-",
        },
        "Per-loop + final-only norm": {
            "ppl": {4: 4.55, 3: 4.56, 2: 4.57, 1: 4.70},
            "color": "#ff7f0e",
            "marker": "^",
            "linestyle": "--",
            "annotate": False,
        },
        "Per-loop + norm penalty": {
            "ppl": {4: 4.53, 3: 4.51, 2: 4.52, 1: 4.67},
            "color": "#d62728",
            "marker": "D",
            "linestyle": ":",
            "annotate": False,
        },
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for label, cfg in curves.items():
        depths = [4, 3, 2, 1]
        xs = [throughputs[k] for k in depths]
        ys = [cfg["ppl"][k] for k in depths]
        ax.plot(
            xs,
            ys,
            marker=cfg["marker"],
            markersize=8,
            linewidth=2.4,
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            label=label,
        )
        if cfg.get("annotate", True):
            for k, x, y in zip(depths, xs, ys):
                dx = -34 if k == 1 else 6
                dy = -10 if label == "Per-loop + raw" else -12
                ax.annotate(f"K={k}", (x, y), textcoords="offset points", xytext=(dx, dy), color=cfg["color"], fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Throughput (tokens/sec)")
    ax.set_ylabel("Validation perplexity (lower is better)")
    ax.set_title("Compute-quality frontier on per-loop-loss 129M models")
    ax.set_ylim(4.48, 5.38)
    ax.set_xlim(19_000, 90_000)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_compute_quality.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "pareto_compute_quality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
