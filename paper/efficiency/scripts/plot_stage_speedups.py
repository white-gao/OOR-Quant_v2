#!/usr/bin/env python3
"""Plot collected RTN measurements / RecQuant estimates; never read mock values."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from plot_speedup_mockup import make_figure, plt


def figure_data(result):
    if result.get("data_kind") != "measured_rtn_and_operator_estimated_recquant":
        raise ValueError("Expected collected measurements/operator estimates, not illustrative data.")
    rows = result.get("rows", [])
    if len(rows) != 12:
        raise ValueError("Expected all twelve figure entries.")
    panels = []
    selected = []
    for stage, key, settings, label in (
        ("prefill", "input_length", result["prefill_lengths"], "Input length (tokens)"),
        ("decode", "batch", result["decode_batches"], "Request batch size"),
    ):
        panel = {"stage": stage.title(), "x_values": settings, "x_label": label, "speedups": {}}
        panel_rows = {}
        for method, kind in (("RTN", "measured_stage"), ("RecQuant", "operator_projected_stage")):
            ordered = []
            for setting in settings:
                matches = [row for row in rows if row["stage"] == stage and row["method"] == method and row[key] == setting]
                if len(matches) != 1 or matches[0]["data_kind"] != kind:
                    raise ValueError(f"Missing/ambiguous or mislabeled {stage}/{method}/{setting}.")
                row = matches[0]
                if not math.isfinite(row["speedup"]) or row["speedup"] <= 0:
                    raise ValueError("Invalid speedup.")
                ordered.append(row)
            panel["speedups"][method] = [row["speedup"] for row in ordered]
            panel_rows[method] = ordered
        panels.append(panel)
        selected.append(panel_rows)
    return {"panels": panels}, selected


def render(source: Path):
    result = json.loads(source.read_text())
    data, selected = figure_data(result)
    fig = make_figure(data)
    try:
        max_ci = max((row["ci95_high"] for row in result["rows"] if row.get("ci95_high") is not None), default=0)
        for ax, panel in zip(fig.axes, selected):
            ax.set_ylim(0, max(ax.get_ylim()[1], max_ci + .15))
            for method_index, (method, offset) in enumerate((("RTN", -.21), ("RecQuant", .21))):
                for index, row in enumerate(panel[method]):
                    if row.get("ci95_low") is not None:
                        x = index + offset
                        ax.vlines(x, row["ci95_low"], row["ci95_high"], color="#253745", linewidth=.7, zorder=7)
                        ax.hlines([row["ci95_low"], row["ci95_high"]], x-.04, x+.04, color="#253745", linewidth=.7, zorder=7)
                        ax.texts[method_index * 3 + index].set_y(max(row["speedup"], row["ci95_high"]) + .035)
        caption = ("OneRec-1.7B W8A8; BF16 reference. RTN stage timings are measured. "
                   "RecQuant stage timings are operator-based estimates, not deployed end-to-end measurements. "
                   "Intervals condition on measured operator means and exclude estimator bias.")
        if result.get("smoke_test"):
            caption = "SMOKE TEST ONLY. " + caption
        stem = "stage_speedups_estimated" + ("_smoke" if result.get("smoke_test") else "")
        for suffix in ("png", "pdf", "svg"):
            target = source.parent / f"{stem}.{suffix}"
            meta = {"Subject": caption} if suffix == "pdf" else {"Description": caption}
            fig.savefig(target, dpi=300, facecolor="white", metadata=meta)
        (source.parent / f"{stem}.caption.txt").write_text(caption + "\n")
        print(f"[figure] {source.parent / (stem + '.pdf')}", flush=True)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    render(parser.parse_args().summary)
