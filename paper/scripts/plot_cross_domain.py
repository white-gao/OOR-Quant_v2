#!/usr/bin/env python3
"""Plot the complete RecQuant transfer matrix from verified evaluation results.

Rows are calibration domains; columns are test domains. The default four-panel
layout shows absolute Pass/Recall at K=16/32, colored by each metric's retention
relative to the corresponding test-domain diagonal. Metrics are not averaged.
Use --layout recall32 to reproduce the first, retention-only preview.
This is a CPU-only preview/export and does not modify the paper's LaTeX files.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ("ad", "product", "video")
LABELS = ("AD", "Product", "Video")
METRICS = ("pass@16", "pass@32", "recall@16", "recall@32")
COUNTS = {"ad": 27677, "product": 27910, "video": 38781}
DEFAULT_SOURCE = ROOT / (
    "artifacts/results/flat_quant/cross_domain/"
    "1p7b_w8a8_pc_recquant_matrix/cross_domain_matrix.json"
)


def load_matrix(source):
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (payload.get("model"), payload.get("precision"), payload.get("method")) != (
        "OneRec-1.7B", "W8A8 per-channel", "RecQuant"
    ):
        raise ValueError("Expected the OneRec-1.7B W8A8 per-channel RecQuant matrix")
    values = np.full((len(METRICS), 3, 3), np.nan)
    seen = set()
    for row in payload["rows"]:
        source_domain, target = row["calibration_domain"], row["test_domain"]
        pair = (source_domain, target)
        if pair in seen or row["status"] not in ("complete", "reused"):
            raise ValueError(f"Duplicate or incomplete matrix entry: {pair}")
        seen.add(pair)
        path = Path(row["result_path"])
        if not path.is_absolute():
            path = ROOT / path
        raw = json.loads(path.read_text(encoding="utf-8"))["1.7B"][target]["test"]
        if raw["total_samples"] != COUNTS[target]:
            raise ValueError(f"Expected the full {target} test split: {path}")
        for m, metric in enumerate(METRICS):
            value = 100 * raw[metric]
            if not np.isclose(value, row[metric], rtol=0, atol=1e-10):
                raise ValueError(f"Summary does not match raw {metric}: {path}")
            values[m, DOMAINS.index(source_domain), DOMAINS.index(target)] = value
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Expected all 36 positive, finite metric values")
    diagonal = np.diagonal(values, axis1=1, axis2=2)
    retention = 100 * values / diagonal[:, None, :]
    for row in payload["rows"]:
        i, j = DOMAINS.index(row["calibration_domain"]), DOMAINS.index(row["test_domain"])
        for m, metric in enumerate(METRICS):
            if not np.isclose(retention[m, i, j], row[f"retention_{metric}_pct"], rtol=0, atol=1e-10):
                raise ValueError("Retention must be normalized by the target-domain diagonal")
    return values, retention


def export_figure(fig, output, *, tight=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = output.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight" if tight else None,
                    pad_inches=0.06, facecolor="white")
        print(path)
    plt.close(fig)


def draw_metrics(values, retention, output):
    """A true 3.5-inch-wide figure with 36 absolute values and one color scale."""
    norm = Normalize(vmin=90, vmax=101)
    if np.any((retention < norm.vmin) | (retention > norm.vmax)):
        raise ValueError("Data exceed the declared 90--101% color range")
    cmap = LinearSegmentedColormap.from_list(
        "transfer_retention", ["#e8b69d", "#f1e5cd", "#c8ded5", "#78aaa4"], N=256
    )
    style = {
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": "#22333d", "axes.labelcolor": "#22333d",
        "xtick.color": "#22333d", "ytick.color": "#22333d",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(3.5, 3.05), facecolor="white")
        positions = ((0.22, 0.61), (0.63, 0.61), (0.22, 0.25), (0.63, 0.25))
        for m, ((left, bottom), metric) in enumerate(zip(positions, METRICS)):
            ax = fig.add_axes([left, bottom, 0.35, 0.29])
            im = ax.imshow(retention[m], cmap=cmap, norm=norm,
                           interpolation="nearest", aspect="auto")
            ax.set_title(metric.capitalize(), fontsize=8.2, fontweight="semibold", pad=4)
            ax.set_xticks(range(3), LABELS)
            ax.set_yticks(range(3), LABELS)
            ax.tick_params(axis="both", which="major", length=0, pad=3, labelsize=7)
            ax.tick_params(labelbottom=m >= 2, labelleft=m % 2 == 0)
            ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.5)
            ax.tick_params(which="minor", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{values[m, i, j]:.3f}", ha="center", va="center",
                            color="#22333d", fontsize=7.2)
                    if i == j:
                        ax.add_patch(Rectangle((j - 0.455, i - 0.445), 0.91, 0.89,
                                               fill=False, linewidth=0.6, edgecolor="#22333d"))
        fig.text(0.60, 0.99, "Values: metric (%)  |  outlined: in-domain",
                 ha="center", va="top", fontsize=6.6)
        fig.text(0.60, 0.166, "Test domain", ha="center", fontsize=8)
        fig.text(0.025, 0.575, "Calibration domain", ha="center", va="center",
                 rotation=90, fontsize=8)
        cax = fig.add_axes([0.22, 0.10, 0.76, 0.022])
        bar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[90, 95, 100])
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=2, width=0.6, labelsize=6.5, pad=2)
        bar.set_label("Color: retention vs. in-domain (%)", fontsize=7, labelpad=2)
        export_figure(fig, output)


def draw(retention, output):
    # A single linear scale for all cells. 100.08% is deliberately nearly the
    # same color as 100%; there is no per-column rescaling or clipped >100 value.
    norm = Normalize(vmin=90, vmax=101)
    if np.any((retention < norm.vmin) | (retention > norm.vmax)):
        raise ValueError("Data exceed the declared 90--101% color range")
    cmap = LinearSegmentedColormap.from_list(
        "transfer_retention", ["#e8b69d", "#f1e5cd", "#c8ded5", "#347c79"], N=256
    )
    style = {
        "font.family": "DejaVu Sans", "font.size": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": "#22333d", "axes.labelcolor": "#22333d",
        "xtick.color": "#22333d", "ytick.color": "#22333d",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(3.6, 4.0), facecolor="white")
        ax = fig.add_axes([0.24, 0.29, 0.70, 0.63])
        im = ax.imshow(retention, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Test domain", labelpad=10)
        ax.set_ylabel("Calibration domain", labelpad=9)
        ax.tick_params(axis="both", which="major", length=0, pad=7)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=3)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(3):
            for j in range(3):
                color = "white" if retention[i, j] >= 99 else "#22333d"
                ax.text(j, i, f"{retention[i, j]:.2f}%", ha="center", va="center",
                        color=color, fontsize=10.5,
                        fontweight="semibold" if i == j else "normal")
                if i == j:
                    ax.add_patch(Rectangle((j - 0.455, i - 0.455), 0.91, 0.91,
                                           fill=False, linewidth=0.85, edgecolor="#22333d"))
        cax = fig.add_axes([0.24, 0.17, 0.70, 0.030])
        bar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[90, 95, 100])
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=2.5, width=0.6, labelsize=7.5)
        bar.set_label("Recall@32 retention (%)", fontsize=9, labelpad=4)
        fig.text(0.59, 0.055, "Outlined cells: in-domain references", ha="center", fontsize=7.5)
        fig.text(0.59, 0.017, "OneRec-1.7B  |  W8A8", ha="center", fontsize=7.5, color="#64737b")
        export_figure(fig, output, tight=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--layout", choices=("metrics", "recall32"), default="metrics")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    values, retention = load_matrix(args.source)
    print("Rows=calibration, columns=test [AD, Product, Video]; no metric averaging.")
    for m, metric in enumerate(METRICS):
        print(f"{metric} (%):\n{np.array2string(values[m], precision=6)}")
        print(f"Retention (%):\n{np.array2string(retention[m], precision=6)}")
    output = args.output or ROOT / "paper/figure" / (
        "cross_domain_metrics" if args.layout == "metrics" else "cross_domain_retention"
    )
    if args.layout == "metrics":
        draw_metrics(values, retention, output)
    else:
        draw(retention[METRICS.index("recall@32")], output)


if __name__ == "__main__":
    main()
