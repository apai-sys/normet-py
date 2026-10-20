"""Figures for print (normet.figstyle): formulas written as chemistry writes
them, a figure laid out at a journal's column width in its lettering and put back as it
was, its panels lettered, what the journal's guide asks read from the figure as drawn,
and the file written at its size exactly with its fonts embedded."""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from normet import figstyle as FS  # noqa: E402
from normet.figstyle import MM, PRESETS  # noqa: E402
from normet.style import formula, plain  # noqa: E402


@pytest.mark.parametrize(
    "name, written",
    [
        ("NO2", r"$\mathregular{NO_{2}}$"),
        ("SO42-", r"$\mathregular{SO_{4}^{2-}}$"),
        ("NH4+", r"$\mathregular{NH_{4}^{+}}$"),
        ("PM2.5", r"$\mathregular{PM_{2.5}}$"),
        ("Meteorologically normalised NO2", r"Meteorologically normalised $\mathregular{NO_{2}}$"),
        # What only reads as a formula: thermal fractions, nothing to set lower, a name.
        ("OC1", "OC1"),
        ("CO", "CO"),
        ("ws", "ws"),
        ("atmos_pres", "atmos_pres"),
        (r"$x_2$", r"$x_2$"),
    ],
)
def test_a_formula_is_written_as_chemistry_writes_it(name, written):
    assert formula(name) == written


@pytest.mark.parametrize(
    "name, quoted",
    [("SO42-", "SO₄²⁻"), ("NO2", "NO₂"), ("PM2.5", "PM₂.₅"), ("OC1", "OC1"), (r"$x_2$", "x₂")],
)
def test_a_formula_is_quoted_as_it_reads(name, quoted):
    assert plain(formula(name)) == quoted


def _figure(rows=2, cols=1, size=(12, 8)):
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.bar(range(4), [1, 2, 3, 4], label="NO2")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["NO2", "O3", "OC", "EC"])
        ax.set_title(f"Site {i + 1}")
        ax.set_ylabel("Concentration")
        ax.text(0.5, 0.5, "a note", fontsize=6, transform=ax.transAxes)
    axes[0, 0].legend()
    fig.suptitle("Normalised")
    return fig, axes


def _ticks_kw(fig):
    return [
        (dict(a._major_tick_kw), dict(a._minor_tick_kw))
        for ax in fig.axes
        for a in (ax.xaxis, ax.yaxis)
    ]


def test_a_figure_is_laid_out_at_its_print_size_and_put_back():
    fig, axes = _figure()
    ax = axes[0, 0]
    before = (
        tuple(fig.get_size_inches()),
        ax.title.get_fontsize(),
        ax.yaxis.label.get_fontsize(),
        [t.get_text() for t in ax.texts],
        fig._suptitle.get_visible(),
        _ticks_kw(fig),
    )
    nature = PRESETS["nature"]
    put_back = FS.to_print(fig, nature, 89, labels="a")
    fig.draw_without_rendering()
    assert np.allclose(fig.get_size_inches()[0] * MM, 89)
    assert ax.title.get_fontsize() == nature.title and ax.yaxis.label.get_fontsize() == nature.text
    assert ax.get_legend().get_texts()[0].get_text() == formula("NO2")
    note = [t for t in ax.texts if t.get_text() == "a note"][0]
    assert nature.min_text <= note.get_fontsize() <= nature.max_text
    assert [t.get_text() for t in ax.get_xticklabels()][0] == formula("NO2")
    assert not fig._suptitle.get_visible()  # a caption's, not the figure's
    face = FS.available_face(nature.fonts)
    assert face in ax.yaxis.label.get_fontfamily()
    put_back()
    fig.draw_without_rendering()
    after = (
        tuple(fig.get_size_inches()),
        ax.title.get_fontsize(),
        ax.yaxis.label.get_fontsize(),
        [t.get_text() for t in ax.texts],
        fig._suptitle.get_visible(),
        _ticks_kw(fig),
    )
    assert after == before  # the axes' tick settings too: none written over theirs
    assert [t.get_text() for t in ax.get_xticklabels()] == ["NO2", "O3", "OC", "EC"]
    plt.close(fig)


def test_a_legend_wider_than_the_column_takes_fewer_columns():
    fig, ax = plt.subplots(figsize=(12, 3))
    for name in ("Observed", "Uncertainty band", "Normalised (deweathered)", "Seeded"):
        ax.plot(range(10), np.arange(10.0), label=name)
    ax.legend(ncols=4, loc="upper center")
    put_back = FS.to_print(fig, PRESETS["nature"], 89)
    renderer = FS._renderer(fig)
    assert ax.get_legend()._ncols < 4
    assert ax.get_legend().get_window_extent(renderer).width <= fig.bbox.width
    put_back()
    assert ax.get_legend()._ncols == 4
    plt.close(fig)


def test_panels_are_lettered_in_reading_order_each_column_in_a_line():
    fig, axes = _figure(2, 2)
    axes[1, 0].set_yticks([0, 2, 4], ["0", "2", "4"])  # narrower labels than above
    made = FS.label_panels(fig, "(a)", bold=False, size=8)
    assert [t.get_text() for t in made] == ["(a)", "(b)", "(c)", "(d)"]
    assert [t.axes for t in made] == [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    x = [t.get_window_extent(renderer).x0 for t in made]
    assert abs(x[0] - x[2]) < 0.5 and abs(x[1] - x[3]) < 0.5
    assert FS.panel_letters(3, "A", start="c") == ["C", "D", "E"]
    plt.close(fig)


def test_a_colour_bar_a_legend_key_and_a_blank_cell_are_not_panels():
    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], width_ratios=[60, 6, 1, 26])
    heat = fig.add_subplot(gs[0, 0:2])
    cax = fig.add_subplot(gs[0, 2])
    series = fig.add_subplot(gs[1, 0:2])
    key = fig.add_subplot(gs[1, 3])
    key.axis("off")
    key.legend(["one", "two"])
    blank = fig.add_subplot(gs[0, 3])
    blank.axis("off")
    im = heat.imshow(np.abs(np.random.default_rng(0).normal(size=(8, 4))))
    fig.colorbar(im, cax=cax)
    assert FS.panel_axes(fig) == [heat, series]
    made = FS.label_panels(fig, "a")
    assert [t.get_text() for t in made] == ["a", "b"]
    plt.close(fig)


def test_the_titles_of_the_figure_and_its_panels_follow_one_switch():
    fig, axes = _figure()
    put_back = FS.to_print(fig, PRESETS["elsevier"], 90)
    assert not fig._suptitle.get_visible()
    assert not any(ax.title.get_visible() for ax in axes.ravel())
    put_back()
    assert fig._suptitle.get_visible()
    assert all(ax.title.get_visible() for ax in axes.ravel())
    axes[0, 0].title.set_visible(False)  # one the figure had left off
    put_back = FS.to_print(fig, PRESETS["elsevier"], 90, title=True)
    assert all(ax.title.get_visible() for ax in axes.ravel())
    put_back()
    assert not axes[0, 0].title.get_visible()  # put back as the figure had it
    plt.close(fig)


def test_the_height_rises_until_each_panel_has_room_for_its_data():
    fig, _ = _figure(4, 1, size=(15, 4))  # four rows drawn wide and low
    FS.to_print(fig, PRESETS["elsevier"], 90)
    renderer = FS._renderer(fig)
    low = min(a.get_window_extent(renderer).height / fig.dpi * MM for a in FS.panel_axes(fig))
    assert low >= FS.MIN_PANEL - 0.5
    plt.close(fig)


def _found(findings, about):
    return [f for f in findings if f.about == about][0]


def test_the_checks_read_the_guide_from_the_figure_as_drawn():
    fig, axes = _figure()
    fig.set_size_inches(200 / MM, 100 / MM)
    for ax in axes.ravel():
        ax.tick_params(labelsize=4)
    found = FS.preflight(fig, PRESETS["elsevier"])
    assert not _found(found, "width").ok and "190 mm" in _found(found, "width").said
    assert not _found(found, "text").ok and "4.0 pt" in _found(found, "text").said
    plt.close(fig)
    # Nature's 7 pt at most leaves a subscript at 4.9 pt, below its 5 pt least.
    fig, _ = _figure()
    FS.to_print(fig, PRESETS["nature"], 183)
    found = FS.preflight(fig, PRESETS["nature"])
    assert not _found(found, "script").ok and "4.9 pt" in _found(found, "script").said
    # The text is quoted as it reads, not as the mathtext that draws it.
    assert "'NO₂'" in _found(found, "script").said and "$" not in _found(found, "script").said
    assert _found(found, "largest text").ok and _found(found, "height").ok
    plt.close(fig)
    # ES&T asks for four panels at most.
    fig, _ = _figure(1, 5)
    assert not _found(FS.preflight(fig, PRESETS["acs"]), "panels").ok
    plt.close(fig)


def test_tick_labels_that_run_into_one_another_are_found():
    fig, ax = plt.subplots(figsize=(30 / MM, 30 / MM))
    ax.plot(range(24), range(24))
    ax.set_xticks(range(24))
    found = _found(FS.preflight(fig, PRESETS["copernicus"]), "tick labels")
    assert not found.ok and "the x axis of panel a" in found.said
    plt.close(fig)


def test_saved_at_its_size_exactly_its_fonts_embedded(tmp_path):
    from PIL import Image

    fig, _ = _figure()
    FS.to_print(fig, PRESETS["acs"], 84.7, 70, labels="(a)")
    pdf = FS.save_for_print(fig, str(tmp_path / "f.pdf"), PRESETS["acs"])
    with open(pdf, "rb") as fh:
        data = fh.read()
    found = re.search(rb"/MediaBox \[ *0 0 ([\d.]+) ([\d.]+)", data)
    assert found is not None
    box = [float(v) for v in found.groups()]
    assert np.allclose(box, [84.7 / MM * 72, 70 / MM * 72], atol=0.01)
    assert b"/FontFile2" in data  # TrueType, Type 42
    with open(FS.save_for_print(fig, str(tmp_path / "f.svg")), encoding="utf-8") as fh:
        svg = fh.read()
    assert "<text" in svg  # its text kept as text
    tif = FS.save_for_print(fig, str(tmp_path / "f.tiff"), PRESETS["acs"])
    with Image.open(tif) as im:
        assert round(im.info["dpi"][0]) == PRESETS["acs"].dpi
        assert abs(im.size[0] - round(84.7 / MM * PRESETS["acs"].dpi)) <= 1
    plt.close(fig)
