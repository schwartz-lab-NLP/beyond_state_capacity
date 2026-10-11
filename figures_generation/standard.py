"""Shared plotting style, colors, and layout helpers for the paper figures.

Import this module from each figure script (`causality.py`, `correlation.py`,
`failure.py`, ...) so that fonts, colors, sizing, and axis conventions stay in
sync across the whole paper.
"""

from enum import Enum
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from libs.paths import RAW_RESULTS_PATH, figure_pdf_path


def output_path(name: str) -> Path:
    """Return the shared output path `$FIGURES_PATH/<name>.pdf`."""
    return figure_pdf_path(name)


class MODELS(str, Enum):
    MAMBA_CODESTRAL = "mamba_codestral"
    MAMBA_NVIDIA = "mamba_nvidia"

    def __str__(self) -> str:
        return self.value

    @property
    def display_name(self) -> str:
        return DISPLAY_NAME[self]


DISPLAY_NAME = {
    MODELS.MAMBA_CODESTRAL: "Codestral-Mamba",
    MODELS.MAMBA_NVIDIA: "Nvidia-Mamba",
}


def parse_model(model: str) -> MODELS:
    """Return the MODELS member for ``model``, or raise ValueError."""
    try:
        return MODELS(model)
    except ValueError:
        valid = ", ".join(m.value for m in MODELS)
        raise ValueError(
            f"Invalid model {model!r}. Must be one of: {valid}"
        ) from None


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

FIG_WIDTH = 5.5

PANEL_LABEL_SIZE = 7.5
MODEL_TITLE_SIZE = 6.5
AXIS_LABEL_SIZE = 6.5
TICK_SIZE = 4.0
BAR_LABEL_SIZE = 5.0
LEGEND_SIZE = 6.5
LINE_WIDTH = 0.8
MARKER_SIZE = 1.2
MARKER_EVERY = 1

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COLOR_LOCAL = "#f09f75"
COLOR_GLOBAL = "#4274b5"
COLOR_DELTA = "#57ad77"
COLOR_REPRESENTATION = "#7e6baf"
COLOR_BAR = "#bfd7ef"
COLOR_MARKER_LAYER = "#F987C5"
COLOR_LAYER_ACCENT = "#bfd7ef"

# Shared 1-indexed layer axis so vertical grid lines line up across figures.
LAYER_XRANGE = {
    MODELS.MAMBA_CODESTRAL: {
        "xlim": (1, 64),
        "xticks": [1, 10, 20, 30, 40, 50, 60],
    },
    MODELS.MAMBA_NVIDIA: {
        "xlim": (1, 56),
        "xticks": [1, 10, 20, 30, 40, 50],
    },
}

ENTRIES_XRANGE = {
    MODELS.MAMBA_CODESTRAL: {
        "xlim": (20, 500),
        "xticks": [100, 200, 300, 400, 500],
    },
    MODELS.MAMBA_NVIDIA: {
        "xlim": (20, 200),
        "xticks": [20, 50, 100, 150, 200],
    },
}

# Inclusive layer span to shade as a vertical background accent.
LAYER_ACCENT = {
    MODELS.MAMBA_CODESTRAL: (10, 33),
    MODELS.MAMBA_NVIDIA: (10, 30),
}

# Shared y-limits for model-accuracy axes (not probe / other metrics).
MODEL_ACC_YLIM = {
    MODELS.MAMBA_CODESTRAL: (0.48, 1.02),
    MODELS.MAMBA_NVIDIA: (-0.02, 1.02),
}

# Canonical per-series styling (color/linestyle/label), keyed consistently
# across figures so that plots and legends line up visually.
SERIES = {
    "mamba_global": {"label": "Unflagged", "color": COLOR_GLOBAL, "ls": "-"},
    "mamba_local": {"label": "Flagged", "color": COLOR_LOCAL, "ls": "-"},
}

LEGEND_SERIES_ORDER = ("mamba_local", "mamba_global")


# ---------------------------------------------------------------------------
# rcParams / axes styling
# ---------------------------------------------------------------------------

def apply_paper_style():
    """Apply the shared matplotlib rcParams (and seaborn theme) used by all
    paper figures. Call once per figure, before creating any axes."""
    font_rc = {
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }
    plt.rcParams.update({
        **font_rc,
        "font.size": AXIS_LABEL_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": MODEL_TITLE_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.linewidth": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    sns.set_theme(style="whitegrid", rc=plt.rcParams)
    # Seaborn's theme resets font.family to sans-serif; restore paper fonts.
    plt.rcParams.update(font_rc)


def style_axes(ax):
    """Apply shared tick styling to a single axes."""
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_SIZE,
        length=2.5,
        width=0.5,
        pad=1.5,
        direction="out",
        labelrotation=0,
    )
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.08):
    """Draw a bold panel label (e.g. "(a)") above the top-left of an axes."""
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top", ha="left",
        clip_on=False,
    )


def apply_shared_layer_axis(ax, model: MODELS, *, show_xlabel: bool):
    """Apply the shared 1-indexed layer x-axis (range + ticks) to an axes."""
    ax.set_xlim(LAYER_XRANGE[model]["xlim"][0]-0.5, LAYER_XRANGE[model]["xlim"][1]+0.5)
    ax.set_xticks(LAYER_XRANGE[model]["xticks"])
    if show_xlabel:
        ax.set_xlabel("Layer")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def apply_layer_accent(ax, model: MODELS):
    """Shade a model-specific layer interval as a light vertical band."""
    xmin, xmax = LAYER_ACCENT[model]
    ax.axvspan(
        xmin, xmax,
        facecolor=COLOR_LAYER_ACCENT,
        edgecolor="none",
        alpha=0.18,
        zorder=0,
    )

def apply_shared_entries_axis(ax, model: MODELS):
    """Apply the shared entries x-axis (range + ticks) to an axes."""
    ax.set_xlim(*ENTRIES_XRANGE[model]["xlim"])
    ax.set_xticks(ENTRIES_XRANGE[model]["xticks"])


def apply_model_acc_ylim(ax, model: MODELS):
    """Apply the shared model-accuracy y-limits to an axes."""
    ax.set_ylim(*MODEL_ACC_YLIM[model])


# ---------------------------------------------------------------------------
# Canonical-series plotting helpers
# ---------------------------------------------------------------------------

def plot_kwargs(series_key: str, *, with_marker: bool = False, **overrides) -> dict:
    """Return matplotlib `plot` kwargs for a canonical series (see SERIES)."""
    style = SERIES[series_key]
    kwargs = {
        "color": style["color"],
        "linestyle": style["ls"],
        "linewidth": LINE_WIDTH,
    }
    if with_marker:
        kwargs.update({
            "marker": "o",
            "markersize": MARKER_SIZE,
            "markevery": MARKER_EVERY,
            "markeredgewidth": 0.3,
        })
    kwargs.update(overrides)
    return kwargs


def legend_handle(series_key: str, *, with_marker: bool = True) -> Line2D:
    """Return a Line2D legend handle for a canonical series."""
    style = SERIES[series_key]
    kwargs = {"color": style["color"], "linestyle": style["ls"], "linewidth": LINE_WIDTH}
    if with_marker:
        kwargs["marker"] = "o"
        kwargs["markersize"] = MARKER_SIZE
    return Line2D([0], [0], **kwargs)


def legend_handles_and_labels(series_keys, *, with_marker: bool = True):
    """Return (handles, labels) lists for a sequence of canonical series keys."""
    handles = [legend_handle(key, with_marker=with_marker) for key in series_keys]
    labels = [SERIES[key]["label"] for key in series_keys]
    return handles, labels


def in_axes_legend(ax, handles=None, labels=None, *, ncol: int = 1, **overrides):
    """Place a compact in-axes legend in the lower-right corner.

    Framed with a light border and semi-transparent background so it sits
    cleanly over plotted content. Shared by figures whose legends live inside
    the axes (causality, failure); not for figure-level external legends.
    """
    kwargs = {
        "loc": "lower right",
        "ncol": ncol,
        "frameon": True,
        "fancybox": False,
        "edgecolor": "0.8",
        "framealpha": 0.85,
        "fontsize": LEGEND_SIZE,
        "borderpad": 0.3,
        "labelspacing": 0.2,
        "handletextpad": 0.3,
        "handlelength": 1.2,
        "columnspacing": 0.6,
        "borderaxespad": 0.4,
    }
    kwargs.update(overrides)
    if handles is None:
        return ax.legend(**kwargs)
    if labels is None:
        return ax.legend(handles=handles, **kwargs)
    return ax.legend(handles, labels, **kwargs)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

PATCHING_AND_SCALING_ENTRIES = "target_entry"
PATCHING_AND_SCALING_ENTRY_PART = "number"
LOCAL_DATASET_FLAG_ENTRIES = "target"
PATCHING_AND_SCALING_RESULTS_FILENAME = (
    f"patch-{PATCHING_AND_SCALING_ENTRIES}_{PATCHING_AND_SCALING_ENTRY_PART}"
    f"-prompt-{LOCAL_DATASET_FLAG_ENTRIES}-results.json"
)


def patching_and_scaling_dir(model: MODELS) -> str:
    return os.path.join(
        RAW_RESULTS_PATH,
        f"{model}_prefix", "patching_and_scaling", "raw_results",
    )


def cumulative_sweep_rows(data: dict) -> list:
    if "cumulative" in data["sweeps"]:
        return data["sweeps"]["cumulative"]
    return data["sweeps"]["all"]
