#!/usr/bin/env python3
"""Export a compact paper figure, full diagnostic figures, and unchanged statistics.

The paper figure focuses on first-step dispersion, cross-cutoff reversals, and
actual beam-path retention. All slots and samples remain in diagnostics and JSON.
Only CPU aggregation/rendering is performed; no model is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from metrics import KS, GAP_EDGES, GAP_LABELS, bootstrap_mean

HERE = Path(__file__).resolve().parents[1]
COLORS = {"a": "#2878B5", "b": "#C45A35", "c": "#439775"}


def read_records(path):
    if not (path / "COMPLETE.json").is_file():
        raise FileNotFoundError(f"Probe incomplete: {path}; run it before plotting")
    records = [json.loads(p.read_text()) for p in sorted((path / "records").glob("*.json"))]
    complete = json.loads((path / "COMPLETE.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if complete["manifest_sha256"] != fingerprint:
        raise ValueError(f"Completion manifest mismatch in {path}")
    expected_ids = {r["id"] for r in manifest["selection"]["selection"]}
    if len(records) != complete["count"] or {r["id"] for r in records} != expected_ids:
        raise ValueError(f"Record count mismatch in {path}")
    return records


def read_text_reference(path, model):
    path = Path(path).resolve()
    rows = read_records(path)
    meta = json.loads((path / "manifest.json").read_text())
    labels = {"cmrc2018": "CMRC2018 (Chinese)", "wikitext2": "WikiText-2"}
    if meta.get("job") != "text" or meta.get("variant") != "bf16" or meta.get("task") not in labels:
        raise ValueError(f"Unknown or non-BF16 text source: {path}")
    if meta["model"] != model:
        raise ValueError("Text and SID distributions must come from the same BF16 model")
    if not rows or any(not r["positions"] for r in rows):
        raise ValueError(f"Empty text articles in {path}")
    if len({r["document_sha256"] for r in rows}) != len(rows):
        raise ValueError(f"Repeated documents in {path}")
    return rows, {"task": meta["task"], "label": labels[meta["task"]], "path": str(path),
                  "manifest_sha256": hashlib.sha256(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                  "documents": len(rows), "positions": sum(len(r["positions"]) for r in rows)}


def summarize_text_positions(rows, bootstrap, *, field="positions", predicate=lambda p: True):
    """Macro-average within each article, then bootstrap articles, not positions."""
    groups = [[p for p in r.get(field, []) if predicate(p)] for r in rows]
    groups = [g for g in groups if g]

    def stats(values):
        result = bootstrap_mean(values, repeats=bootstrap)
        result["median"] = float(np.median(values)) if values else None
        return result

    result = {"documents": len(groups), "positions": sum(map(len, groups)),
              "mass": {str(k): stats([float(np.mean([p["full_vocab"]["mass"][str(k)] for p in g])) for g in groups]) for k in KS},
              "entropy_nats": stats([float(np.mean([p["full_vocab"]["entropy_nats"] for p in g])) for g in groups])}
    for key in ("target_nll", "top1_correct", "sid_token_mass"):
        values = [float(np.mean([p[key] for p in g])) for g in groups if all(key in p for p in g)]
        result[key] = stats(values)  # Missing legacy metrics remain null, not zero.
    return result


def export_text_controls(references, teacher, out, task, bootstrap, plt):
    """Supplement: all text sources/slots, paired length controls, health statistics."""
    bins = [("128-255", 128, 255), ("256-511", 256, 511),
            ("512-1024", 512, 1024), (">1024", 1025, float("inf"))]
    summary = {"bootstrap_unit": "article / user", "references": {}, "sid_length_bins": {}}
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.5))
    for index, (rows, meta) in enumerate(references):
        key = meta["task"]
        if key in summary["references"]:
            raise ValueError(f"Repeated text reference: {key}")
        stats = summarize_text_positions(rows, bootstrap)
        stats["source"] = meta
        stats["length_bins"] = {
            label: summarize_text_positions(rows, bootstrap, predicate=lambda p, lo=lo, hi=hi: lo <= p["prefix_tokens"] <= hi)
            for label, lo, hi in bins}
        controls = sorted({p["prefix_tokens"] for r in rows for p in r.get("length_control", [])})
        stats["paired_length_control"] = {
            str(n): summarize_text_positions(rows, bootstrap, field="length_control", predicate=lambda p, n=n: p["prefix_tokens"] == n)
            for n in controls}
        summary["references"][key] = stats
        draw_summary_band(a, KS, [stats["mass"][str(k)] for k in KS], label=meta["label"],
                          color=["#555555", "#9467bd"][index % 2], style="--")
        if controls:
            for k, color in ((1, COLORS["a"]), (32, COLORS["b"])):
                draw_summary_band(b, controls, [stats["paired_length_control"][str(n)]["mass"][str(k)] for n in controls],
                                  label=f"{meta['label']} top-{k}", color=color)
    for slot_index, slot in enumerate(("a", "b", "c")):
        arrays = [[r["slots"][slot]["full_vocab"]["mass"][str(k)] for r in teacher] for k in KS]
        stats = [bootstrap_mean(values, repeats=bootstrap) for values in arrays]
        draw_summary_band(a, KS, stats, label=f"SID-{slot.upper()}", color=COLORS[slot])
        summary["sid_length_bins"][slot] = {}
        for label, lo, hi in bins:
            values = [r["slots"][slot]["full_vocab"]["mass"]["32"] for r in teacher
                      if lo <= r["prompt_tokens"] + slot_index <= hi]
            summary["sid_length_bins"][slot][label] = bootstrap_mean(values, repeats=bootstrap)
    a.set(xscale="log", xlabel="Top-k candidates", ylabel="Cumulative mass (%)",
          ylim=(0, 102), title="Text sources and all SID positions")
    a.set_xticks(KS, [str(k) for k in KS])
    a.legend(fontsize=7)
    b.set(xlabel="Context length (tokens)", ylabel="Cumulative mass (%)", ylim=(0, 102),
          title="Same positions, different context lengths")
    b.set_xticks([256, 512, 1024])
    if b.lines:
        b.legend(fontsize=7)
    else:
        b.text(.5, .5, "Length control not collected", transform=b.transAxes, ha="center")
    for ax in (a, b):
        ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"text_controls_{task}.{ext}", dpi=250, bbox_inches="tight")
    (out / f"text_controls_{task}.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def means_band(ax, xs, arrays, label, color, style="-", bootstrap=2000):
    stats = [bootstrap_mean(a, repeats=bootstrap) for a in arrays]
    mean = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
    low = [s["low"] if s["low"] is not None else np.nan for s in stats]
    high = [s["high"] if s["high"] is not None else np.nan for s in stats]
    ax.plot(xs, mean, style, color=color, label=label, linewidth=1.5, marker=".")
    ax.fill_between(xs, low, high, color=color, alpha=.10)
    return stats


def boundary_heatmap_data(summary):
    """Drop only bins empty in every row; missing cells are not zero reversals."""
    rows = [(variant, slot) for variant in ("w8a8", "w4a8") for slot in ("a", "b", "c")]
    rates = np.array([
        [np.nan if stat["mean"] is None else 100 * stat["mean"]
         for stat in summary["conditional"][variant][slot]["gap_bins"]]
        for variant, slot in rows
    ])
    columns = np.flatnonzero(np.isfinite(rates).any(axis=0))
    if not len(columns):
        raise ValueError("No populated boundary-gap bins to plot")
    return rows, columns, rates[:, columns]


def draw_summary_band(ax, xs, stats, *, label, color, style="-", marker="o"):
    """Render already-computed query/document means and bootstrap CIs in percent."""
    values = np.array([[s[key] for s in stats] for key in ("mean", "low", "high")], dtype=float) * 100
    ax.plot(xs, values[0], linestyle=style, marker=marker, markersize=3.5,
            linewidth=1.5, color=color, label=label, zorder=3)
    ax.fill_between(xs, values[1], values[2], color=color, alpha=.16, linewidth=0)


def paper_figure(summary, plt):
    """Three-panel, double-column layout; no additional statistical selection."""
    from matplotlib.ticker import PercentFormatter

    with plt.rc_context({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.25, 2.55),
                                      gridspec_kw={"width_ratios": [1, 1.30, 1]})
        fig.subplots_adjust(left=.065, right=.99, top=.85, bottom=.24, wspace=.48)

        # The claim in this panel is explicitly about the first SID decision.
        # Full conditional B/C distributions remain in diagnostics_<task>.*.
        text_label = summary.get("text_reference", {}).get("label", "Chinese text")
        for slot, label, color in (("text", text_label, "#666666"),
                                   ("a", "SID-A", COLORS["a"])):
            stats = [summary["concentration"][slot]["mass"][str(k)] for k in KS]
            draw_summary_band(a, KS, stats, label=label, color=color)
            mass32 = summary["concentration"][slot]["mass"]["32"]["mean"] * 100
            offset = (4, 6) if slot == "text" else (5, -12)
            a.annotate(f"{mass32:.1f}%", (32, mass32), xytext=offset,
                       textcoords="offset points", fontsize=7, color=color)
        a.axvline(32, color="#aaaaaa", linestyle=":", linewidth=.8, zorder=0)
        a.set(xscale="log", xlabel="Top-k candidates", ylabel="Cumulative probability mass",
              ylim=(0, 102), xlim=(.85, 170), title="(a) First-step dispersion")
        a.set_xticks([1, 4, 16, 32, 128], ["1", "4", "16", "32", "128"])
        a.set_xticks([], minor=True)
        a.set_yticks([0, 25, 50, 75, 100])
        a.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        a.legend(loc="upper left", frameon=False, handlelength=1.2, borderaxespad=.2)

        rows, columns, values = boundary_heatmap_data(summary)
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#f0f0f0")
        # A fixed 0--100% color scale and explicit numbers avoid visual inflation.
        b.imshow(np.ma.masked_invalid(values), vmin=0, vmax=100, cmap=cmap,
                 aspect="auto", interpolation="nearest")
        for row in range(len(rows)):
            for col in range(len(columns)):
                rate = values[row, col]
                label = "—" if np.isnan(rate) else ("<0.1" if 0 < rate < .1 else f"{rate:.1f}")
                b.text(col, row, label, ha="center", va="center", fontsize=6.5,
                       color="white" if rate >= 65 else "#162a3a")

        def gap_tick(i):
            lo, hi = GAP_EDGES[i:i+2]
            compact = lambda x: f"{x:g}".removeprefix("0")
            return f">{lo:g}" if np.isinf(hi) else f"({compact(lo)},{compact(hi)}]"

        b.set_xticks(range(len(columns)), [gap_tick(i) for i in columns],
                     rotation=45, ha="right", rotation_mode="anchor")
        b.set_yticks(range(len(rows)), [f"{v.upper()}  {s.upper()}" for v, s in rows], fontsize=6.6)
        b.set_xticks(np.arange(-.5, len(columns)), minor=True)
        b.set_yticks(np.arange(-.5, len(rows)), minor=True)
        b.grid(which="minor", color="white", linewidth=.8)
        b.tick_params(which="both", length=0)
        b.axhline(2.5, color="white", linewidth=2)
        b.set(xlabel="Teacher logit gap", title="(b) Cross-cutoff reversals (%)")
        for spine in b.spines.values():
            spine.set_visible(False)

        for variant, color, style, marker in (("w8a8", COLORS["a"], "--", "o"),
                                               ("w4a8", COLORS["b"], "-", "s")):
            stats = summary["beam"][variant]["path_retention"]
            draw_summary_band(c, [1, 2, 3], stats, label=variant.upper(), color=color,
                              style=style, marker=marker)
            for depth, stat in enumerate(stats, 1):
                offset = 7 if variant == "w8a8" or depth == 3 else -12
                c.annotate(f"{stat['mean']*100:.1f}", (depth, stat["mean"]*100),
                           xytext=(0, offset), textcoords="offset points", ha="center",
                           color=color, fontsize=6.8)
        c.set(xlabel="Retained SID prefix", ylabel="Beam-path retention",
              xticks=[1, 2, 3], xticklabels=["A", "AB", "ABC"],
              xlim=(.78, 3.25), ylim=(0, 105), title="(c) Complete-path divergence")
        c.set_yticks([0, 25, 50, 75, 100])
        c.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        c.legend(loc="center left", bbox_to_anchor=(0, .53), frameon=False,
                 handlelength=1.5, borderaxespad=.2)
        recall = [summary["beam"][v]["sid_recall"]["32"]["mean"] * 100
                  for v in ("bf16", "w8a8", "w4a8")]
        recall_label = ("Subset SID Recall@32 (%)\n"
                        "BF16      W8A8      W4A8\n"
                        f"{recall[0]:.2f}        {recall[1]:.2f}        {recall[2]:.2f}")
        c.text(.50, .035, recall_label, transform=c.transAxes, fontsize=6.5,
               ha="center", va="bottom", linespacing=1.35,
               bbox={"facecolor": "#f5f6f7", "edgecolor": "none", "pad": 3})
        for ax in (a, c):
            ax.grid(axis="y", color="#dddddd", linewidth=.5, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        return fig


def export_paper_panels(fig, out, task):
    """Separate vector panels for LaTeX subfigures; captions supply (a)/(b)/(c)."""
    from matplotlib.transforms import Bbox

    titles = [ax.get_title() for ax in fig.axes]
    visible = [ax.get_visible() for ax in fig.axes]
    try:
        for ax in fig.axes:
            ax.set_title("")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()) for ax in fig.axes]
        bottom, top = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
        for index, (ax, box) in enumerate(zip(fig.axes, boxes)):
            for other in fig.axes:
                other.set_visible(other is ax)
            crop = Bbox.from_extents(box.x0-.04, bottom-.04, box.x1+.04, top+.04)
            for ext in ("pdf", "png"):
                fig.savefig(out / f"motivation_{task}_{'abc'[index]}.{ext}",
                            dpi=300, bbox_inches=crop, pad_inches=0)
    finally:
        for ax, title, shown in zip(fig.axes, titles, visible):
            ax.set_title(title)
            ax.set_visible(shown)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(HERE / "results/v1_seed42"))
    parser.add_argument("--task", default="ad", choices=("ad", "product", "video"))
    parser.add_argument("--output-dir")
    parser.add_argument("--text-root", help="Completed primary text probe; requires a separate --output-dir")
    parser.add_argument("--extra-text-root", action="append", default=[], help="Completed same-model supplementary text probe")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    if args.text_root and not args.output_dir:
        raise ValueError("Specify a separate --output-dir when replacing the primary text reference")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("Plotting needs matplotlib: install it in your benchmark environment. Probe outputs are unaffected.")
    root = Path(args.run_root).resolve()
    out = Path(args.output_dir).resolve() if args.output_dir else HERE / "figures" / root.name
    if HERE not in out.parents:
        raise ValueError("Figures must be saved under paper/motivation/")
    out.mkdir(parents=True, exist_ok=True)
    teacher = read_records(root / args.task / "conditional/bf16")
    teacher_meta = json.loads((root / args.task / "conditional/bf16/manifest.json").read_text())
    text, text_reference = read_text_reference(Path(args.text_root) if args.text_root else root / "text/bf16", teacher_meta["model"])
    references = [(text, text_reference)] + [read_text_reference(path, teacher_meta["model"]) for path in args.extra_text_root]
    conditional = {v: read_records(root / args.task / "conditional" / v) for v in ("w8a8", "w4a8")}
    beams = {v: read_records(root / args.task / "beam" / v) for v in ("bf16", "w8a8", "w4a8")}
    if any({r["id"] for r in rows} != {r["id"] for r in teacher} for rows in conditional.values()):
        raise ValueError("Conditional query IDs are not paired")
    if any({r["id"] for r in rows} != {r["id"] for r in beams["bf16"]} for rows in beams.values()):
        raise ValueError("Beam query IDs are not paired")
    summary = {"task": args.task, "run_root": str(root), "bootstrap_unit": "user query / text document",
               "conditional_n": len(teacher), "beam_n": len(beams["bf16"]), "text_documents": len(text),
               "concentration": {}, "conditional": {}, "beam": {}, "gap_bins": list(GAP_LABELS),
               "text_reference": text_reference}
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))
    a, b, c, d = axes.flat
    for slot in ("a", "b", "c"):
        arrays = [[r["slots"][slot]["full_vocab"]["mass"][str(k)] for r in teacher] for k in KS]
        stats = means_band(a, KS, arrays, f"SID-{slot.upper()}", COLORS[slot], bootstrap=args.bootstrap)
        summary["concentration"][slot] = {"mass": dict(zip(map(str, KS), stats)),
            "entropy_nats": bootstrap_mean([r["slots"][slot]["full_vocab"]["entropy_nats"] for r in teacher], repeats=args.bootstrap),
            "legal_mass": bootstrap_mean([r["slots"][slot]["legal_mass"] for r in teacher], repeats=args.bootstrap)}
    arrays = [[np.mean([p["full_vocab"]["mass"][str(k)] for p in r["positions"]]) for r in text] for k in KS]
    stats = means_band(a, KS, arrays, text_reference["label"], "#666666", bootstrap=args.bootstrap)
    summary["concentration"]["text"] = {"mass": dict(zip(map(str, KS), stats)),
        "entropy_nats": bootstrap_mean([np.mean([p["full_vocab"]["entropy_nats"] for p in r["positions"]]) for r in text], repeats=args.bootstrap)}
    a.set(xscale="log", xlabel="Candidate rank k", ylabel="Cumulative probability mass",
          ylim=(0, 1.02), title="(a) BF16 candidate concentration")
    a.set_xticks(KS, [str(k) for k in KS])
    a.legend(fontsize=8)
    appendix, nmse_ax = plt.subplots(figsize=(5, 3.8))
    for variant, style, marker in (("w8a8", "--", "o"), ("w4a8", "-", "x")):
        summary["conditional"][variant] = {}
        for slot in ("a", "b", "c"):
            rows = [r["slots"][slot] for r in conditional[variant]]
            label = f"{variant.upper()} / {slot.upper()}"
            b.scatter([r["kl"] for r in rows], [r["top32_retention"] for r in rows],
                      s=8, alpha=.25, color=COLORS[slot], marker=marker, label=label)
            nmse_ax.scatter([r["final_block_nmse"] for r in rows], [r["top32_retention"] for r in rows],
                            s=8, alpha=.25, color=COLORS[slot], marker=marker, label=label)
            gap_arrays = [[r["boundary_bins"][i]["flips"] / r["boundary_bins"][i]["count"]
                           for r in rows if r["boundary_bins"][i]["count"]] for i in range(len(GAP_LABELS))]
            gap_stats = means_band(c, np.arange(len(GAP_LABELS)), gap_arrays, label, COLORS[slot],
                                   style, args.bootstrap)
            stats = {key: bootstrap_mean([r[key] for r in rows], repeats=args.bootstrap) for key in
                     ("kl", "top32_retention", "tie_aware_top32_retention", "top1_agreement", "cutoff_gap",
                      "teacher_zero_gap_pairs", "teacher_near_tie_pairs", "final_block_nmse")}
            stats["gap_bins"] = gap_stats
            stats["gap_pair_counts"] = [sum(r["boundary_bins"][i]["count"] for r in rows) for i in range(len(GAP_LABELS))]
            stats["student_tie_counts"] = [sum(r["boundary_bins"][i]["student_ties"] for r in rows) for i in range(len(GAP_LABELS))]
            summary["conditional"][variant][slot] = stats
        arrays = [[r["path_retention"][str(depth)] for r in beams[variant]
                   if r["path_retention"][str(depth)] is not None] for depth in (1, 2, 3)]
        color = "#2878B5" if variant == "w8a8" else "#C45A35"
        summary["beam"][variant] = {"path_retention": means_band(d, [1, 2, 3], arrays, variant.upper(), color,
                                                                 bootstrap=args.bootstrap)}
    for variant, rows in beams.items():
        summary["beam"].setdefault(variant, {}).update({
            "sid_recall": {str(k): bootstrap_mean([r["sid_recall"][str(k)] for r in rows], repeats=args.bootstrap) for k in (1, 8, 32)},
            "invalid_returned_fraction": bootstrap_mean([r["invalid_returned_fraction"] for r in rows], repeats=args.bootstrap),
            "queries_with_early_stop": sum(r["trace"]["early_stopped_candidates"] > 0 for r in rows)})
        if variant != "bf16":
            summary["beam"][variant]["paired_delta_sid_recall32"] = bootstrap_mean([r["delta_sid_recall32"] for r in rows], repeats=args.bootstrap)
            summary["beam"][variant]["lost_teacher_hits"] = sum(r["lost_teacher_hit32"] for r in rows)
            summary["beam"][variant]["gained_hits"] = sum(r["gained_hit32"] for r in rows)
    b.set(xlabel="Conditional KL (nats)", ylabel="Local Top-32 retention", ylim=(-.03, 1.03),
          title="(b) Distribution and candidate agreement", xscale="symlog")
    b.set_xscale("symlog", linthresh=.001)
    b.set_xlim(left=0)
    b.legend(fontsize=7, ncol=2)
    c.set(xlabel="Teacher logit gap", ylabel="Strict reversal rate (student gap < 0)",
          title="(c) Teacher ranks 1–32 vs. 33–64", ylim=(-.03, 1.03))
    c.set_xticks(range(len(GAP_LABELS)), GAP_LABELS, rotation=30, ha="right")
    _, populated, _ = boundary_heatmap_data(summary)
    c.set_xlim(populated[0] - .3, populated[-1] + .3)
    c.legend(fontsize=7, ncol=2)
    d.set(xlabel="Decoding depth", ylabel="Actual beam-path retention", ylim=(-.03, 1.03),
          title="(d) Free-running beam search (width 32)")
    d.set_xticks([1, 2, 3], ["A", "AB", "ABC"])
    d.legend()
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle(f"OneRec-1.7B / {args.task.upper()} — FlatQuant per-channel", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"diagnostics_{args.task}.pdf", bbox_inches="tight")
    fig.savefig(out / f"diagnostics_{args.task}.png", dpi=200, bbox_inches="tight")
    publication = paper_figure(summary, plt)
    publication.savefig(out / f"motivation_{args.task}.pdf", bbox_inches="tight", pad_inches=.025)
    publication.savefig(out / f"motivation_{args.task}.png", dpi=300, bbox_inches="tight", pad_inches=.025)
    export_paper_panels(publication, out, args.task)
    nmse_ax.set(xlabel="Final-block output NMSE (SID prediction positions)", ylabel="Local Top-32 retention", ylim=(-.03, 1.03))
    nmse_ax.set_xscale("symlog", linthresh=.001)
    nmse_ax.set_xlim(left=0)
    nmse_ax.legend(fontsize=7, ncol=2)
    appendix.tight_layout()
    appendix.savefig(out / f"nmse_retention_{args.task}.pdf", bbox_inches="tight")
    if args.text_root or args.extra_text_root:
        export_text_controls(references, teacher, out, args.task, args.bootstrap, plt)
    (out / f"summary_{args.task}.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plt.close("all")
    print(f"[plots and statistics] {out}")


if __name__ == "__main__":
    main()
