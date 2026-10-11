"""Bar chart of gate-patching and gate-scaling causal interventions."""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import standard
from standard import MODELS, parse_model

FIG_WIDTH = standard.FIG_WIDTH
FIG_HEIGHT = 2.4

BAR_LABEL_SIZE = standard.BAR_LABEL_SIZE
LEGEND_SIZE = standard.LEGEND_SIZE
LINE_WIDTH = standard.LINE_WIDTH

LOCAL_COLOR = standard.COLOR_LOCAL
GLOBAL_COLOR = standard.COLOR_GLOBAL
BAR_COLOR = standard.COLOR_BAR

LAYER_RANGE = {
    MODELS.MAMBA_CODESTRAL: "1-64",
    MODELS.MAMBA_NVIDIA: "1-56",
}

N_PATCHING_BARS = 2
PATCHING_SCALING_SEPARATOR_X = N_PATCHING_BARS - 0.5
SECTION_LABEL_Y_INSET = 0.02
SECTION_LABEL_X_INSET = 0.02


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_experiment_data(model: MODELS, component: str, direction: str) -> dict:
    results_path = os.path.join(
        standard.patching_and_scaling_dir(model),
        component,
        direction,
        standard.PATCHING_AND_SCALING_RESULTS_FILENAME,
    )
    return _load_json(results_path)


def _cumulative_accuracy(data: dict, model: MODELS) -> float:
    """Accuracy for a given layer range in the cumulative sweep."""
    layer_range = LAYER_RANGE[model]
    for row in standard.cumulative_sweep_rows(data):
        if row["layer_range"] == layer_range:
            return row["accuracy"]
    raise KeyError(f"layer_range {layer_range!r} not found in cumulative sweep")


def _load_baselines_and_bars(model: MODELS):
    """Return (local_acc, global_acc, bars) where bars is a list of (label, accuracy)."""
    local_to_global = _load_experiment_data(model, "delta", "local_to_global")
    # Unpatched reference accuracies from the main local→global Δ run.
    local_acc = local_to_global["donor_accuracy"]
    global_acc = local_to_global["recipient_accuracy"]

    global_to_local = _load_experiment_data(model, "delta", "global_to_local")

    # scale_0_5 = _load_experiment_data(model, "delta", "global0.5x")
    scale_0_8 = _load_experiment_data(model, "delta", "global0.8x")
    scale_1_2 = _load_experiment_data(model, "delta", "global1.2x")
    scale_1_5 = _load_experiment_data(model, "delta", "global1.5x")
    scale_2 = _load_experiment_data(model, "delta", "global2x")
    scale_3 = _load_experiment_data(model, "delta", "global3x")

    bars = [
        ("Unflagged ← Flagged\nGates", _cumulative_accuracy(local_to_global, model)),
        ("Flagged ← Unflagged\nGates", _cumulative_accuracy(global_to_local, model)),
        # ("Unflagged Gates\n0.5$\\times$ Scale", _cumulative_accuracy(scale_0_5, model)),
        ("Unflagged Gates\n0.8$\\times$ Scale", _cumulative_accuracy(scale_0_8, model)),
        ("Unflagged Gates\n1.2$\\times$ Scale", _cumulative_accuracy(scale_1_2, model)),
        ("Unflagged Gates\n1.5$\\times$ Scale", _cumulative_accuracy(scale_1_5, model)),
        ("Unflagged Gates\n2$\\times$ Scale", _cumulative_accuracy(scale_2, model)),
        ("Unflagged Gates\n3$\\times$ Scale", _cumulative_accuracy(scale_3, model)),
    ]
    return local_acc, global_acc, bars


def _add_section_panel_labels(ax):
    """Place (a)/(b) at the top corners of the patching and scaling regions."""
    x_left, _ = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    y = ymax - SECTION_LABEL_Y_INSET * (ymax - ymin)
    label_kwargs = {
        "fontsize": standard.PANEL_LABEL_SIZE,
        "fontweight": "bold",
        "va": "top",
        "ha": "left",
        "zorder": 5,
    }
    ax.text(
        x_left + SECTION_LABEL_X_INSET, y, "(a)",
        **label_kwargs,
    )
    ax.text(
        PATCHING_SCALING_SEPARATOR_X + SECTION_LABEL_X_INSET,
        y,
        "(b)",
        **label_kwargs,
    )


def plot(
    ax,
    model: MODELS,
    *,
    show_xticks: bool = True,
    show_legend: bool = True,
    panel_label: str | None = None,
    show_section_labels: bool = False,
):
    """Draw the causality bar chart onto ``ax``."""
    local_acc, global_acc, bars = _load_baselines_and_bars(model)
    labels = [label for label, _ in bars]
    accuracies = [acc for _, acc in bars]

    x = range(len(bars))
    ax.bar(x, accuracies, color=BAR_COLOR, width=0.7, zorder=2)

    ax.axvline(
        PATCHING_SCALING_SEPARATOR_X,
        color="0.55",
        linestyle=":",
        linewidth=LINE_WIDTH,
        zorder=1,
    )

    ax.axhline(
        local_acc, color=LOCAL_COLOR, linestyle="--", linewidth=LINE_WIDTH,
        zorder=3,
    )
    ax.axhline(
        global_acc, color=GLOBAL_COLOR, linestyle="--", linewidth=LINE_WIDTH,
        zorder=3,
    )

    ax.set_xticks(list(x))
    ax.set_ylabel("Model acc.")
    standard.apply_model_acc_ylim(ax, model)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.grid(False, axis="x")
    standard.style_axes(ax)

    if show_xticks:
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", labelsize=BAR_LABEL_SIZE)
    else:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    if show_legend:
        legend_handles = [
            Line2D(
                [0], [0], color=LOCAL_COLOR, linestyle="--",
                linewidth=LINE_WIDTH, label="Flagged baseline",
            ),
            Line2D(
                [0], [0], color=GLOBAL_COLOR, linestyle="--",
                linewidth=LINE_WIDTH, label="Unflagged baseline",
            ),
        ]
        standard.in_axes_legend(ax, handles=legend_handles)

    if panel_label is not None:
        standard.panel_label(ax, panel_label)

    if show_section_labels:
        _add_section_panel_labels(ax)


def main(model: MODELS):
    output_path = standard.output_path(f"{model}_causality")
    standard.apply_paper_style()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plot(ax, model)
    fig.tight_layout(pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", pad_inches=0.05, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate causality figure.")
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
