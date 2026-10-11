"""Three-panel correlation figure: evaluation, state probe, and gate contrast."""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import standard
from standard import MODELS, parse_model
from libs.paths import RAW_RESULTS_PATH

FIG_WIDTH = standard.FIG_WIDTH
LEFT_FRAC = 0.4
# Shorter than square so (a) leaves room above the shared legend.
PLOT_SIZE = FIG_WIDTH * LEFT_FRAC * 0.792
LEGEND_HEIGHT = 0.42
FIG_HEIGHT = PLOT_SIZE + LEGEND_HEIGHT

TICK_SIZE = standard.TICK_SIZE
LEGEND_SIZE = standard.LEGEND_SIZE
LINE_WIDTH = standard.LINE_WIDTH
MARKER_SIZE = standard.MARKER_SIZE
MARKER_EVERY = standard.MARKER_EVERY


def _probe_layers(layer_index) -> np.ndarray:
    """Map 0-indexed probe layer indices onto the shared 1-indexed axis."""
    return np.asarray(layer_index, dtype=int) + 1


def _layer_axis(n_layers: int) -> np.ndarray:
    """Return 1-indexed layer positions for n_layers values."""
    return np.arange(1, n_layers + 1)


# Panel labels sit slightly further left here to clear the taller left panel.
_PANEL_LABEL_X = -0.22


# ---------------------------------------------------------------------------
# Panel 1: Evaluation accuracy vs. number of entries
# ---------------------------------------------------------------------------

def _eval_runs(model: MODELS):
    return [
        {
            "series": "mamba_global",
            "path": os.path.join(
                RAW_RESULTS_PATH,
                f"{model}_prefix", "evaluation", "phonebook",
                "raw_results", "results.json",
            ),
        },
        {
            "series": "mamba_local",
            "path": os.path.join(
                RAW_RESULTS_PATH,
                f"{model}_prefix", "evaluation", "phonebook_local",
                "raw_results", "target_results.json",
            ),
        },
    ]


def plot_evaluation(ax, model: MODELS, *, panel_label_y: float = 1.08):
    for run in _eval_runs(model):
        with open(run["path"]) as f:
            data = json.load(f)
        ax.plot(
            data["lengths"],
            data["accuracy_list"],
            **standard.plot_kwargs(run["series"], with_marker=True),
        )
    ax.set_xlabel("Number of Entries")
    ax.set_ylabel("Model acc.")
    standard.apply_shared_entries_axis(ax, model)
    standard.apply_model_acc_ylim(ax, model)
    ax.grid(True, alpha=0.3)
    standard.style_axes(ax)
    standard.panel_label(ax, "(a)", x=_PANEL_LABEL_X, y=panel_label_y)


# ---------------------------------------------------------------------------
# Panel 2: State-probe accuracy vs. layer
# ---------------------------------------------------------------------------

SHAS = {
    MODELS.MAMBA_NVIDIA: {
        "profbook": "1178",
        "profbook_local": "8e5e",
    },
    MODELS.MAMBA_CODESTRAL: {
        "profbook": "5449",
        "profbook_local": "c3e5",
    },
}


def _probe_runs(model: MODELS):
    return [
        {
            "series": "mamba_local",
            "path": os.path.join(
                RAW_RESULTS_PATH,
                f"{model}_prefix", "state_probing",
                "profbook_local", "raw_results", SHAS[model]["profbook_local"], "probe_results.parquet",
            ),
        },
        {
            "series": "mamba_global",
            "path": os.path.join(
                RAW_RESULTS_PATH,
                f"{model}_prefix", "state_probing",
                "profbook", "raw_results", SHAS[model]["profbook"], "probe_results.parquet",
            ),
        },
    ]


def plot_state_probing(ax, model: MODELS):
    for run in _probe_runs(model):
        df = pd.read_parquet(run["path"])
        series = df.groupby("layer")["probe_success"].mean()
        ax.plot(
            _probe_layers(series.index),
            series.values,
            **standard.plot_kwargs(run["series"], with_marker=True),
        )
    ax.set_ylabel("State Probe acc.")
    standard.apply_shared_layer_axis(ax, model, show_xlabel=False)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    standard.style_axes(ax)
    standard.panel_label(ax, "(b)", x=_PANEL_LABEL_X)


# ---------------------------------------------------------------------------
# Panel 3: dt_raw entry-number contrast by layer (mean across heads)
# ---------------------------------------------------------------------------

_SELECTIVITY_DATASETS = {
    "phonebook": "mamba_global",
    "phonebook_local": "mamba_local",
}


def _load_selectivity_results(model: MODELS, dataset: str) -> dict:
    path = os.path.join(
        RAW_RESULTS_PATH,
        f"{model}_prefix", "selection_distinction",
        dataset, "raw_results", "dt_raw_results.json",
    )
    with open(path) as f:
        data = json.load(f)
    return {key: np.array(val) for key, val in data.items()}


def _mean_entry_number_contrast(results: dict) -> np.ndarray:
    target = results["target_entry_number"]
    other = results["other_entries_number"]
    return target.mean(axis=1) - other.mean(axis=1)


def plot_selection_distinction(ax, model: MODELS):
    for dataset, series_key in _SELECTIVITY_DATASETS.items():
        results = _load_selectivity_results(model, dataset)
        contrast = _mean_entry_number_contrast(results)
        ax.plot(
            _layer_axis(len(contrast)),
            contrast,
            **standard.plot_kwargs(series_key, with_marker=True),
        )
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Gate mean diff")
    standard.apply_shared_layer_axis(ax, model, show_xlabel=True)
    ax.grid(True, alpha=0.3)
    standard.style_axes(ax)
    standard.panel_label(ax, "(c)", x=_PANEL_LABEL_X)


def main(model: MODELS):
    output_path = standard.output_path(f"{model}_correlation")
    standard.apply_paper_style()

    legend_bottom = LEGEND_HEIGHT / FIG_HEIGHT
    plot_bottom = legend_bottom + 0.04
    plot_top = 0.97

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[LEFT_FRAC, 1 - LEFT_FRAC],
        height_ratios=[1, 1],
        wspace=0.35,
        hspace=0.08,  # tighter: no xlabel/tick labels between (b) and (c)
        left=0.11,
        right=0.98,
        top=plot_top,
        bottom=plot_bottom,
    )
    ax_left = fig.add_subplot(gs[:, 0])
    ax_top_right = fig.add_subplot(gs[0, 1])
    ax_bot_right = fig.add_subplot(gs[1, 1], sharex=ax_top_right)

    # (a) spans both rows, so axes-fraction y=1.08 sits higher than on (b).
    # Scale the offset by the height ratio so (a) and (b) share a figure y.
    # Draw first: get_position() can be 0 before the canvas is realized.
    fig.canvas.draw()
    left_height = ax_left.get_position().height
    height_ratio = (
        ax_top_right.get_position().height / left_height if left_height else 0.5
    )
    plot_evaluation(ax_left, model, panel_label_y=1 + 0.08 * height_ratio)
    plot_state_probing(ax_top_right, model)
    plot_selection_distinction(ax_bot_right, model)

    legend_handles, legend_labels = standard.legend_handles_and_labels(
        standard.LEGEND_SERIES_ORDER
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, legend_bottom * 0.45),
        columnspacing=0.5,
        handletextpad=0.25,
        handlelength=1.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", pad_inches=0.05, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close()


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate correlation figure.")
    parser.add_argument(
        "--model",
        default=MODELS.MAMBA_NVIDIA,
        choices=[m.value for m in MODELS],
        help="Model to generate the figure for",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(parse_model(args.model))
