"""Summarize the loss-placement / scale-visibility decomposition.

The mechanism-suite CSV already contains the raw diagnostics.  This script
turns those rows into compact paper artifacts:

* a 2x2 loss-placement-vs-scale table,
* a quadrant figure showing early-exit usability versus norm control, and
* a compact visibility--activity mechanism diagram, and
* a radial-gradient / norm-drift correlation check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


CORE_ORDER = [
    "terminal_norm",
    "terminal_raw",
    "perstep_norm",
    "perstep_raw",
]

SCALE_ORDER = {
    "44m": 0,
    "129m": 1,
}

LABELS = {
    "terminal_norm": "Terminal + RMSNorm",
    "terminal_raw": "Terminal + raw",
    "perstep_norm": "Per-loop + RMSNorm",
    "perstep_raw": "Per-loop + raw",
    "perstep_final_norm": "Per-loop + final-only norm",
    "terminal_norm_penalty": "Terminal + norm penalty",
}

MARKERS = {
    "terminal_norm": "o",
    "terminal_raw": "s",
    "perstep_norm": "^",
    "perstep_raw": "D",
    "perstep_final_norm": "P",
    "terminal_norm_penalty": "X",
}

COLORS = {
    "terminal_norm": "#8c6d31",
    "terminal_raw": "#1f77b4",
    "perstep_norm": "#d62728",
    "perstep_raw": "#2ca02c",
    "perstep_final_norm": "#9467bd",
    "terminal_norm_penalty": "#17becf",
}

READOUT_COLORS = {
    "norm": "#d55e00",
    "raw": "#0072b2",
}

LOSS_MARKERS = {
    "terminal": "o",
    "per-step": "^",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mechanism-csv",
        default="outputs/mechanism_suite_44m_129m_s42/mechanism_metrics.csv",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/endpoint_scale_story",
    )
    p.add_argument(
        "--paper-fig-dir",
        default="paper_v5/figures",
        help="Optional figure copy target for LaTeX. Use '' to skip.",
    )
    p.add_argument(
        "--paper-table-dir",
        default="paper_v5/tables",
        help="Optional table copy target for LaTeX. Use '' to skip.",
    )
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted: dict[str, Any] = {}
            for key, value in row.items():
                if key in {
                    "params",
                    "ppl_K1",
                    "ppl_K4",
                    "norm_K1",
                    "norm_K4",
                    "final_scale_ce_range",
                    "radial_cos_loop1",
                    "radial_cos_loop4",
                    "num_tokens",
                    "norm_penalty_weight",
                }:
                    converted[key] = float(value)
                elif key in {
                    "use_decode_norm",
                    "decode_norm_final_only",
                    "use_spectral_damping",
                }:
                    converted[key] = value == "True"
                else:
                    converted[key] = value
            converted["scale"] = converted["checkpoint"].split("_", 1)[0]
            converted["condition"] = converted["checkpoint"].split("_", 1)[1]
            converted["endpoint_trained"] = converted["supervision"] == "per-step"
            converted["scale_visible"] = (
                not converted["use_decode_norm"]
                or converted["decode_norm_final_only"]
                or converted["norm_penalty_weight"] > 0
            )
            converted["ce_K1"] = math.log(converted["ppl_K1"])
            converted["ce_K4"] = math.log(converted["ppl_K4"])
            converted["early_exit_ce_gap"] = converted["ce_K1"] - converted["ce_K4"]
            converted["log10_norm_K4"] = math.log10(converted["norm_K4"])
            converted["log10_radial_K4"] = math.log10(
                max(converted["radial_cos_loop4"], 1e-12)
            )
            rows.append(converted)
    return rows


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "checkpoint",
        "scale",
        "condition",
        "endpoint_trained",
        "scale_visible",
        "ppl_K1",
        "ppl_K4",
        "early_exit_ce_gap",
        "norm_K4",
        "radial_cos_loop4",
        "log10_norm_K4",
        "log10_radial_K4",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def sci(value: float) -> str:
    if value == 0:
        return "0"
    exp = math.floor(math.log10(abs(value)))
    coeff = value / (10**exp)
    return rf"${coeff:.1f}{chr(92)}times10^{{{exp}}}$"


def fmt_ppl(value: float) -> str:
    if value > 1e6:
        return sci(value)
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.2f}"


def fmt_gap(value: float) -> str:
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:.2f}"


def write_tex(rows: list[dict[str, Any]], path: Path) -> None:
    core = [row for row in rows if row["condition"] in CORE_ORDER]
    core.sort(
        key=lambda row: (
            SCALE_ORDER.get(row["scale"], 99),
            CORE_ORDER.index(row["condition"]),
        )
    )
    lines = [
        r"\begin{tabular}{llccrrrr}",
        r"\toprule",
        r"\textbf{Scale} & \textbf{Condition} & \textbf{Per-loop CE} & \textbf{Scale-visible} & \textbf{K=1 PPL} & \textbf{K=4 PPL} & \textbf{CE gap} & \textbf{$r_K$} \\",
        r"\midrule",
    ]
    last_scale = None
    for row in core:
        if last_scale is not None and row["scale"] != last_scale:
            lines.append(r"\midrule")
        endpoint = r"\checkmark" if row["endpoint_trained"] else "--"
        visible = r"\checkmark" if row["scale_visible"] else "--"
        lines.append(
            " & ".join(
                [
                    row["scale"].upper(),
                    LABELS[row["condition"]],
                    endpoint,
                    visible,
                    fmt_ppl(row["ppl_K1"]),
                    fmt_ppl(row["ppl_K4"]),
                    fmt_gap(row["early_exit_ce_gap"]),
                    sci(row["radial_cos_loop4"]),
                ]
            )
            + r" \\"
        )
        last_scale = row["scale"]
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    core = [row for row in rows if row["condition"] in CORE_ORDER]
    all_diagnostic = [
        row
        for row in rows
        if row["condition"]
        in set(CORE_ORDER + ["perstep_final_norm", "terminal_norm_penalty"])
    ]
    summary: dict[str, Any] = {}
    for name, selected in {
        "core_2x2": core,
        "core_plus_controls": all_diagnostic,
    }.items():
        xs = [row["log10_radial_K4"] for row in selected]
        ys = [row["log10_norm_K4"] for row in selected]
        summary[name] = {
            "n": len(selected),
            "pearson_log_radial_vs_log_norm": pearson(xs, ys),
            "spearman_log_radial_vs_log_norm": pearson(rank(xs), rank(ys)),
            "mean_ce_gap_by_endpoint_trained": {
                str(flag): mean(
                    row["early_exit_ce_gap"]
                    for row in selected
                    if row["endpoint_trained"] == flag
                )
                for flag in [False, True]
            },
            "mean_log10_norm_by_scale_visible": {
                str(flag): mean(
                    row["log10_norm_K4"]
                    for row in selected
                    if row["scale_visible"] == flag
                )
                for flag in [False, True]
            },
        }
    return summary


def plot_quadrant(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    core = [row for row in rows if row["condition"] in CORE_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(6.55, 4.15), sharey=True)
    x_threshold = 1.0
    y_threshold = 2.0
    x_min = -0.45
    x_max = max(row["early_exit_ce_gap"] for row in core) + 8.0
    y_min = max(0.65, min(row["log10_norm_K4"] for row in core) - 0.35)
    y_max = max(row["log10_norm_K4"] for row in core) + 0.45
    desired_ymax_frac = (y_threshold - y_min) / (y_max - y_min)

    def row_style(row: dict[str, Any]) -> tuple[str, str]:
        readout = "raw" if row["condition"].endswith("_raw") else "norm"
        loss = "per-step" if row["condition"].startswith("perstep") else "terminal"
        return READOUT_COLORS[readout], LOSS_MARKERS[loss]

    def condition_row(scale_rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
        return next(row for row in scale_rows if row["condition"] == condition)

    def draw_arrow(
        ax: Any,
        start: dict[str, Any],
        end: dict[str, Any],
        label: str | None = None,
        label_offset: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        ax.annotate(
            "",
            xy=(end["early_exit_ce_gap"], end["log10_norm_K4"]),
            xytext=(start["early_exit_ce_gap"], start["log10_norm_K4"]),
            arrowprops={
                "arrowstyle": "->",
                "color": "#555555",
                "lw": 0.95,
                "alpha": 0.80,
                "shrinkA": 9,
                "shrinkB": 9,
                "mutation_scale": 10,
            },
            zorder=2,
        )
        if label:
            mid_x = 0.5 * (start["early_exit_ce_gap"] + end["early_exit_ce_gap"])
            mid_y = 0.5 * (start["log10_norm_K4"] + end["log10_norm_K4"])
            ax.text(
                mid_x + label_offset[0],
                mid_y + label_offset[1],
                label,
                fontsize=7.2,
                color="#444444",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                },
                zorder=4,
            )

    def draw_horizontal_ce_arrow(
        ax: Any,
        start: dict[str, Any],
        end: dict[str, Any],
        label: str | None = None,
    ) -> None:
        y = end["log10_norm_K4"] - 0.30
        x_end = max(0.10, end["early_exit_ce_gap"] + 0.10)
        ax.annotate(
            "",
            xy=(x_end, y),
            xytext=(start["early_exit_ce_gap"], y),
            arrowprops={
                "arrowstyle": "->",
                "color": "#555555",
                "lw": 0.95,
                "alpha": 0.80,
                "shrinkA": 7,
                "shrinkB": 7,
                "mutation_scale": 10,
            },
            zorder=2,
        )
        if label:
            ax.text(
                5.0,
                y + 0.14,
                label,
                fontsize=7.2,
                color="#444444",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                },
                zorder=4,
            )

    for ax, scale in zip(axes, ["44m", "129m"]):
        scale_rows = [row for row in core if row["scale"] == scale]

        # Blue marks usable exits, orange marks controlled scale; their overlap is
        # the desired region that satisfies both requirements.
        ax.axvspan(x_min, x_threshold, color="#dbeafe", alpha=0.48, zorder=0)
        ax.axhspan(y_min, y_threshold, color="#ffedd5", alpha=0.50, zorder=0)
        ax.axvspan(x_min, x_threshold, ymin=0.0, ymax=desired_ymax_frac, color="#dcfce7", alpha=0.66, zorder=0.2)
        ax.axvline(x_threshold, color="#777777", linestyle="--", linewidth=0.9, zorder=1)
        ax.axhline(y_threshold, color="#777777", linestyle="--", linewidth=0.9, zorder=1)
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.15)

        terminal_norm = condition_row(scale_rows, "terminal_norm")
        terminal_raw = condition_row(scale_rows, "terminal_raw")
        perstep_norm = condition_row(scale_rows, "perstep_norm")
        perstep_raw = condition_row(scale_rows, "perstep_raw")

        # Highlight the target condition with a halo behind the actual data point.
        ax.scatter(
            perstep_raw["early_exit_ce_gap"],
            perstep_raw["log10_norm_K4"],
            s=245,
            marker="o",
            facecolor="#dcfce7",
            edgecolor="#16a34a",
            linewidth=1.4,
            alpha=0.90,
            zorder=2.6,
        )

        for row in scale_rows:
            color, marker = row_style(row)
            ax.scatter(
                row["early_exit_ce_gap"],
                row["log10_norm_K4"],
                s=82,
                marker=marker,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
            )

        corner_style = {
            "fontsize": 6.7,
            "ha": "center",
            "va": "center",
            "bbox": {
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.58,
            },
            "zorder": 1.4,
        }
        ax.text(0.21, 0.88, "usable exits\n+ drift", transform=ax.transAxes, color="#1d4ed8", **corner_style)
        ax.text(0.76, 0.88, "bad exits\n+ drift", transform=ax.transAxes, color="#7c2d12", **corner_style)
        ax.text(0.78, 0.10, "bad exits\n+ controlled", transform=ax.transAxes, color="#9a3412", **corner_style)
        ax.text(
            0.22,
            0.105,
            "desired:\nusable + controlled",
            transform=ax.transAxes,
            fontsize=8.7,
            color="#166534",
            ha="center",
            va="center",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "#f0fdf4",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=1.8,
        )
        ax.text(
            x_threshold * 1.12,
            y_threshold + 0.28,
            "usable exits:\nCE gap <= 1",
            fontsize=6.8,
            color="#666666",
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
        )
        ax.text(
            3.0,
            y_threshold + 0.05,
            r"$\|H_K\|_2 \leq 10^2$",
            fontsize=7,
            color="#666666",
            va="bottom",
        )

        ax.set_title(scale.upper())
        ax.set_xlabel("Early-exit failure: CE(1) - CE(4) (symlog)")
        ax.set_xlim(x_min, x_max)
        ax.set_xticks([0, 1, 5, 20])
        ax.set_xticklabels(["0", "1", "5", "20"])
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel(r"Norm drift: $\log_{10}\|H_K\|_2$")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=READOUT_COLORS["norm"],
               markeredgecolor="black", label="RMSNorm readout", markersize=7),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=READOUT_COLORS["raw"],
               markeredgecolor="black", label="raw readout", markersize=7),
        Line2D([0], [0], marker=LOSS_MARKERS["terminal"], color="#555555",
               linestyle="none", label="terminal CE", markersize=7),
        Line2D([0], [0], marker=LOSS_MARKERS["per-step"], color="#555555",
               linestyle="none", label="per-loop CE", markersize=7),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    paths = [
        out_dir / "endpoint_scale_quadrant.pdf",
        out_dir / "endpoint_scale_quadrant.png",
    ]
    for path in paths:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_visibility_activity(out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.35, 2.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(
        xy: tuple[float, float],
        wh: tuple[float, float],
        text: str,
        facecolor: str,
        edgecolor: str,
        fontsize: float = 9.0,
        weight: str = "normal",
    ) -> None:
        patch = patches.FancyBboxPatch(
            xy,
            wh[0],
            wh[1],
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(
            xy[0] + wh[0] / 2,
            xy[1] + wh[1] / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        color: str = "#555555",
        lw: float = 1.4,
        style: str = "->",
        rad: float = 0.0,
    ) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": style,
                "color": color,
                "lw": lw,
                "connectionstyle": f"arc3,rad={rad}",
                "shrinkA": 3,
                "shrinkB": 3,
            },
        )

    box((0.04, 0.40), (0.15, 0.20), r"$H_k=s_k u_k$", "#f8fafc", "#334155", 10, "bold")

    ax.text(0.03, 0.79, "readout path", fontsize=9, fontweight="bold", color="#1d4ed8")
    box((0.26, 0.67), (0.17, 0.17), "output RMSNorm/\nLayerNorm readout", "#dbeafe", "#2563eb", 8.1)
    box((0.52, 0.67), (0.14, 0.17), "direction\n$u_k$", "#eff6ff", "#2563eb", 9.2, "bold")
    box((0.75, 0.67), (0.17, 0.17), "CE loss", "#eff6ff", "#2563eb", 10, "bold")
    arrow((0.19, 0.50), (0.26, 0.75), "#2563eb")
    arrow((0.43, 0.755), (0.52, 0.755), "#2563eb")
    arrow((0.66, 0.755), (0.75, 0.755), "#2563eb")
    ax.text(
        0.48,
        0.88,
        r"CE radial gradient $\approx 0$",
        fontsize=9,
        color="#1d4ed8",
        ha="center",
    )

    ax.text(0.03, 0.23, "recurrent path", fontsize=9, fontweight="bold", color="#9a3412")
    box((0.26, 0.16), (0.21, 0.18), r"$F(H_k)=H_k+B(\mathrm{Norm}(H_k))$", "#ffedd5", "#ea580c", 7.8)
    box((0.55, 0.16), (0.20, 0.18), r"$H_{k+1}=s_k u_k+b(u_k)$", "#fff7ed", "#ea580c", 8.5, "bold")
    box((0.80, 0.16), (0.17, 0.18), r"$s_{k+1}\approx s_k+a_{\mathrm{rad}}(u_k)$", "#fef3c7", "#d97706", 7.8)
    arrow((0.19, 0.50), (0.26, 0.25), "#ea580c")
    arrow((0.47, 0.25), (0.55, 0.25), "#ea580c")
    arrow((0.75, 0.25), (0.80, 0.25), "#ea580c")
    ax.text(
        0.54,
        0.05,
        "scale is carried forward and can drift",
        fontsize=9,
        color="#9a3412",
        ha="center",
    )

    ax.text(
        0.115,
        0.35,
        "same hidden state",
        fontsize=8,
        color="#475569",
        ha="center",
    )
    fig.tight_layout(pad=0.25)
    paths = [
        out_dir / "visibility_activity_mechanism.pdf",
        out_dir / "visibility_activity_mechanism.png",
    ]
    for path in paths:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_radial_norm(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    selected = [
        row
        for row in rows
        if row["condition"]
        in set(CORE_ORDER + ["perstep_final_norm", "terminal_norm_penalty"])
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    for row in selected:
        ax.scatter(
            row["log10_radial_K4"],
            row["log10_norm_K4"],
            s=72,
            marker=MARKERS.get(row["condition"], "o"),
            color=COLORS.get(row["condition"], "#444444"),
            edgecolor="black",
            linewidth=0.5,
        )
        ax.annotate(
            row["checkpoint"].replace("_", " "),
            (row["log10_radial_K4"], row["log10_norm_K4"]),
            xytext=(5, 2),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel(r"$\log_{10}$ final radial-gradient fraction")
    ax.set_ylabel(r"$\log_{10}$ final hidden norm")
    ax.set_title("Radial signal anticorrelates with norm drift")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    paths = [
        out_dir / "radial_norm_correlation.pdf",
        out_dir / "radial_norm_correlation.png",
    ]
    for path in paths:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(Path(args.mechanism_csv))
    table_path = out_dir / "story_table.tex"
    write_csv(rows, out_dir / "story_metrics.csv")
    write_tex(rows, table_path)
    summary = summarize(rows)
    (out_dir / "story_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures = (
        plot_quadrant(rows, out_dir)
        + plot_visibility_activity(out_dir)
        + plot_radial_norm(rows, out_dir)
    )

    if args.paper_fig_dir:
        fig_dir = Path(args.paper_fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for path in figures:
            if path.suffix == ".pdf":
                target = fig_dir / path.name
                target.write_bytes(path.read_bytes())
    if args.paper_table_dir:
        table_dir = Path(args.paper_table_dir)
        table_dir.mkdir(parents=True, exist_ok=True)
        (table_dir / "endpoint_scale_story.tex").write_text(table_path.read_text())

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
