"""CPU-only checks for the explicitly synthetic figure preview."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("speedup_mockup", ROOT / "scripts/plot_speedup_mockup.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def test_clean_preview_scale_and_label_spacing():
    data = plot.load_mockup(ROOT / "mockup/illustrative_speedups.json")
    fig = plot.make_figure(data)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert tuple(fig.get_size_inches()) == pytest.approx((3.5, 1.95))
        assert not fig.texts
        assert len(fig.axes) == 2
        legend_labels = [text.get_text() for legend in fig.legends for text in legend.get_texts()]
        assert legend_labels == ["RTN", "RecQuant", "BF16 = 1×"]
        assert [ax.get_title() for ax in fig.axes] == ["(a) Prefill", "(b) Decode"]
        assert [ax.get_xlabel() for ax in fig.axes] == ["Input tokens", "Request batch"]
        assert fig.axes[0].get_ylabel() == "Speedup (×)"
        labels = list(fig.legends)
        for ax, panel in zip(fig.axes, data["panels"]):
            assert ax.get_ylim()[0] == 0
            assert len(ax.patches) == 6
            assert len(ax.texts) == 6  # Only the six bar values remain.
            numbers = ax.texts
            assert [text.get_text() for text in numbers] == [
                f"{value:.2f}" for method in plot.METHODS for value in panel["speedups"][method]
            ]
            labels.extend([ax.title, ax.xaxis.label, *numbers,
                           *ax.get_xticklabels(), *ax.get_yticklabels()])
            for index, left in enumerate(numbers):
                for right in numbers[index + 1:]:
                    assert not left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer))
        labels.append(fig.axes[0].yaxis.label)
        for artist in labels:
            bounds = artist.get_window_extent(renderer)
            assert bounds.x0 >= 0 and bounds.y0 >= 0
            assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
        for ax in fig.axes:
            assert not fig.legends[0].get_window_extent(renderer).overlaps(ax.title.get_window_extent(renderer))
    finally:
        plot.plt.close(fig)


@pytest.mark.parametrize("kind", ["measured", None])
def test_cannot_present_measured_data_as_this_preview(tmp_path, kind):
    data = json.loads((ROOT / "mockup/illustrative_speedups.json").read_text())
    data["data_kind"] = kind
    path = tmp_path / "source.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="illustrative"):
        plot.load_mockup(path)


def test_nonfinite_speedup_rejected(tmp_path):
    data = json.loads((ROOT / "mockup/illustrative_speedups.json").read_text())
    data["panels"][0]["speedups"]["RTN"][0] = float("nan")
    path = tmp_path / "source.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="finite"):
        plot.load_mockup(path)


def test_missing_comparison_rejected(tmp_path):
    data = json.loads((ROOT / "mockup/illustrative_speedups.json").read_text())
    del data["panels"][1]["speedups"]["RecQuant"]
    path = tmp_path / "source.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="RTN and RecQuant"):
        plot.load_mockup(path)
