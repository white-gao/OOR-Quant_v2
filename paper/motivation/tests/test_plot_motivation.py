"""CPU-only checks for the compact paper figure and empty-bin handling."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from metrics import KS
from plot_motivation import boundary_heatmap_data, export_paper_panels, paper_figure


def stat(value):
    if value is None:
        return {"mean": None, "low": None, "high": None, "n": 0}
    return {"mean": value, "low": max(0, value - .01),
            "high": min(1, value + .01), "n": 10}


def example_summary():
    summary = {"concentration": {}, "conditional": {}, "beam": {}}
    for slot in ("a", "b", "c", "text"):
        summary["concentration"][slot] = {
            "mass": {str(k): stat(.1 + .1*i) for i, k in enumerate(KS)}}
    for variant in ("w8a8", "w4a8"):
        summary["conditional"][variant] = {}
        for slot in ("a", "b", "c"):
            rates = [None, None, None if slot == "a" else .4, .3, .2, .1, .0002, 0]
            summary["conditional"][variant][slot] = {"gap_bins": [stat(x) for x in rates]}
        summary["beam"][variant] = {
            "path_retention": [stat(.9), stat(.7), stat(.4)],
            "sid_recall": {"32": stat(.05)}}
    summary["beam"]["bf16"] = {"sid_recall": {"32": stat(.08)}}
    return summary


def test_heatmap_omits_only_globally_empty_bins():
    summary = example_summary()
    before = deepcopy(summary)
    rows, columns, rates = boundary_heatmap_data(summary)
    assert rows == [(v, s) for v in ("w8a8", "w4a8") for s in "abc"]
    assert columns.tolist() == [2, 3, 4, 5, 6, 7]
    assert rates.shape == (6, 6)
    assert np.isnan(rates[0, 0]) and np.isnan(rates[3, 0])
    assert rates[1, 0] == pytest.approx(40)
    assert np.all(rates[:, -1] == 0)  # Populated zero-reversal bins must remain.
    assert summary == before


def test_all_empty_gap_bins_are_reported():
    summary = example_summary()
    for slots in summary["conditional"].values():
        for slot in slots.values():
            slot["gap_bins"] = [stat(None) for _ in range(8)]
    with pytest.raises(ValueError, match="No populated"):
        boundary_heatmap_data(summary)


def test_paper_figure_has_three_panels_and_preserves_statistics():
    summary = example_summary()
    before = deepcopy(summary)
    fig = paper_figure(summary, plt)
    try:
        assert len(fig.axes) == 3
        assert fig.axes[1].images[0].get_clim() == (0, 100)
        np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(), [90, 70, 40])
        labels = [t.get_text() for t in fig.axes[1].texts]
        assert "—" in labels and "0.0" in labels and "<0.1" in labels
        for fmt, signature in (("png", b"\x89PNG"), ("pdf", b"%PDF-")):
            output = BytesIO()
            fig.savefig(output, format=fmt, bbox_inches="tight")
            assert output.getvalue().startswith(signature)
        assert summary == before
    finally:
        plt.close(fig)


def test_separate_panels_preserve_figure_and_source(tmp_path):
    summary = example_summary()
    before = deepcopy(summary)
    fig = paper_figure(summary, plt)
    titles = [ax.get_title() for ax in fig.axes]
    try:
        export_paper_panels(fig, tmp_path, "ad")
        for panel in "abc":
            assert (tmp_path / f"motivation_ad_{panel}.pdf").read_bytes().startswith(b"%PDF-")
            assert (tmp_path / f"motivation_ad_{panel}.png").read_bytes().startswith(b"\x89PNG")
        assert [ax.get_title() for ax in fig.axes] == titles
        assert all(ax.get_visible() for ax in fig.axes)
        assert summary == before
    finally:
        plt.close(fig)
