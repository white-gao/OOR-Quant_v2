#!/usr/bin/env python3
"""CPU-only figure layout preview using synthetic data.

This does not benchmark a model or predict speedups. A future measured-data
plotter must read timing records; it must not promote these values to results.
The clean layout has no on-figure disclaimer; provenance is retained in the
source JSON, exported metadata, mockup filenames, and EXPERIMENT_PLAN.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("RTN", "RecQuant")
STAGES = ("Prefill", "Decode")
COLORS = {"RTN": "#799AB6", "RecQuant": "#D97950"}
X_LABELS = {"Prefill": "Input tokens", "Decode": "Request batch"}
WARNING = "SYNTHETIC DATA — NOT MEASURED"


def load_mockup(source: Path) -> dict:
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema_version") != 3 or data.get("data_kind") != "illustrative":
        raise ValueError("This plotter accepts only explicitly illustrative data.")
    if data.get("reference") != "BF16" or data.get("precision") != "W8A8":
        raise ValueError("Expected a W8A8 layout normalized to BF16.")
    if not data.get("warning") or data.get("model") != "OneRec-1.7B" or data.get("beam_width") != 32:
        raise ValueError("Expected a synthetic-data warning and the 1.7B, beam-32 preview.")
    panels = data.get("panels", [])
    if [panel.get("stage") for panel in panels] != list(STAGES):
        raise ValueError("Expected Prefill and Decode panels, in that order.")
    for panel in panels:
        x_values = panel.get("x_values", [])
        if len(x_values) != 3 or any(type(value) is not int or value <= 0 for value in x_values):
            raise ValueError("Every stage needs three positive integer workload settings.")
        if not panel.get("x_label") or not panel.get("fixed_setting"):
            raise ValueError("Every stage needs axis and fixed-setting labels.")
        speedups = panel.get("speedups", {})
        if set(speedups) != set(METHODS):
            raise ValueError("Every stage must compare RTN and RecQuant.")
        for method in METHODS:
            values = speedups[method]
            if len(values) != len(x_values) or any(
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in values
            ):
                raise ValueError("Every stage/method needs positive finite speedups for all settings.")
    return data


def make_figure(data: dict):
    style = {
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 6.5,
        "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "text.color": "#253745",
        "axes.labelcolor": "#253745", "xtick.color": "#253745",
        "ytick.color": "#253745",
    }
    with plt.rc_context(style):
        # Draw at the intended single-column size instead of shrinking a wide
        # canvas (which also halves the effective font size in the paper).
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95), sharey=True)
        fig.subplots_adjust(left=.14, right=.99, bottom=.23, top=.76, wspace=.12)
        maximum = max(value for panel in data["panels"]
                      for method in METHODS for value in panel["speedups"][method])
        ymax = max(1.6, math.ceil((maximum + .15) / .2) * .2)
        for index, (ax, panel) in enumerate(zip(axes, data["panels"])):
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E6EBEF", linewidth=.6)
            for method, offset in zip(METHODS, (-.21, .21)):
                positions = [setting + offset for setting in range(len(panel["x_values"]))]
                values = panel["speedups"][method]
                bars = ax.bar(positions, values, width=.36,
                              color=COLORS[method], edgecolor="white", linewidth=.5, zorder=3)
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + .035,
                            f"{value:.2f}", ha="center", va="bottom", fontsize=5.2,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": .15},
                            zorder=6)
            ax.axhline(1, color="#435462", linewidth=.9, linestyle=(0, (4, 3)), zorder=4)
            ax.set_ylim(0, ymax)
            ax.set_xlim(-.55, 2.55)
            ax.set_xticks(range(3), panel["x_values"])
            ax.set_xlabel(X_LABELS[panel["stage"]], labelpad=3)
            ax.set_title(f"({chr(97 + index)}) {panel['stage']}", pad=5, fontweight="semibold")
            ax.yaxis.set_major_locator(MultipleLocator(.4))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
            ax.tick_params(axis="both", length=2, width=.5, pad=2)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#A8B3BB")
            ax.spines["left"].set_color("#A8B3BB")
            if index:
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", left=False)
        axes[0].set_ylabel("Speedup (×)", labelpad=3)
        handles = [Patch(facecolor=COLORS[method], label=method) for method in METHODS]
        handles.append(Line2D([0], [0], color="#435462", linestyle=(0, (4, 3)),
                              linewidth=.9, label="BF16 = 1×"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, 1.0),
                   ncol=3, frameon=False, fontsize=7, handlelength=1.2,
                   handletextpad=.4, columnspacing=.9, borderaxespad=.2)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "mockup/illustrative_speedups.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    data = load_mockup(args.source)
    fig = make_figure(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = args.output_dir / f"inference_speedup_mockup.{extension}"
        fig.savefig(path, dpi=250, facecolor="white",
                    metadata={"Title": "Inference speedup layout preview", "Subject": WARNING}
                    if extension == "pdf" else {"Description": WARNING})
        print(f"[SYNTHETIC PREVIEW] {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
