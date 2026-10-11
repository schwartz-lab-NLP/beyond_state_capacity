"""Two-panel failure figure: patching/scaling and representation probes."""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import standard
from standard import MODELS, parse_model
from libs.paths import RAW_RESULTS_PATH

FIG_WIDTH = standard.FIG_WIDTH
FIG_HEIGHT = 4.6

AXIS_LABEL_SIZE = standard.AXIS_LABEL_SIZE
TICK_SIZE = standard.TICK_SIZE
LEGEND_SIZE = standard.LEGEND_SIZE
LINE_WIDTH = standard.LINE_WIDTH
MARKER_SIZE = standard.MARKER_SIZE
MARKER_EVERY = standard.MARKER_EVERY

MARKER_LAYER_COLOR = standard.COLOR_MARKER_LAYER

LOCAL_BASELINE_COLOR = standard.COLOR_LOCAL
GLOBAL_BASELINE_COLOR = standard.COLOR_GLOBAL

SWEEPS = ("isolated", "cumulative")
DEFAULT_SWEEP = "cumulative"


def _patch_components(model: MODELS) -> dict:
    patching_and_scaling_dir = standard.patching_and_scaling_dir(model)
    return {
        "delta": {
            "label": r"Gate Patching",
            "color": standard.COLOR_DELTA,
            "ls": "-",
            "path": os.path.join(
                patching_and_scaling_dir,
                "delta",
                "local_to_global",
                standard.PATCHING_AND_SCALING_RESULTS_FILENAME
            ),
        },
        "u": {
            "label": "Representation Patching",
            "color": standard.COLOR_REPRESENTATION,
            "ls": "-",
            "path": os.path.join(
                patching_and_scaling_dir,
                "u",
                "local_to_global",
                standard.PATCHING_AND_SCALING_RESULTS_FILENAME
            ),
        },
    }


def _baseline_path(model: MODELS) -> str:
    return os.path.join(
        standard.patching_and_scaling_dir(model),
        "delta",
        "local_to_global",
        standard.PATCHING_AND_SCALING_RESULTS_FILENAME
    )


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _plot_kwargs_probe(series_key: str) -> dict:
    return standard.plot_kwargs(
        series_key, with_marker=True, label=standard.SERIES[series_key]["label"],
    )


def _probe_layer(layer_key: str) -> int:
    """Map 0-indexed probe layer keys onto the shared 1-indexed axis."""
    return int(layer_key) + 1


def _layer_index(layer_range: str) -> int:
    """Parse isolated sweep label into a 1-indexed layer."""
    return int(layer_range)


def _cumulative_layer_end(layer_range: str) -> int:
    """Parse cumulative sweep label ``1-k`` into end layer ``k``."""
    if "-" in layer_range:
        return int(layer_range.split("-")[1])
    return int(layer_range)


def _configure_yaxis(ax, *, ylabel: str, show_ylabel: bool, ytick_right: bool):
    if show_ylabel:
        ax.set_ylabel(ylabel)
    if ytick_right:
        ax.yaxis.tick_right()
        ax.tick_params(
            axis="y", which="both",
            left=False, right=True, labelleft=False, labelright=True,
        )


def _add_layer_marker(ax, marker_layer: int | None):
    if marker_layer is None:
        return
    ax.axvline(
        marker_layer,
        color=MARKER_LAYER_COLOR,
        linestyle=":",
        linewidth=LINE_WIDTH,
        alpha=0.95,
        zorder=4,
    )
    ymin, ymax = ax.get_ylim()
    y_pos = ymax - 0.04 * (ymax - ymin)
    ax.text(
        marker_layer + 0.4,
        y_pos,
        f"layer {marker_layer}",
        fontsize=AXIS_LABEL_SIZE,
        color=MARKER_LAYER_COLOR,
        va="top",
        ha="left",
        clip_on=True,
    )

# ---------------------------------------------------------------------------
# Panel (a): Representation probe AUC vs. layer
# ---------------------------------------------------------------------------

def _representation_probe_runs(model: MODELS):
    return [
        {
            "series": "mamba_local",
            "path": os.path.join(
                RAW_RESULTS_PATH,
                f"{model}_prefix", "representation_probing", "phonebook_local",
                "raw_results", "target_results.json",
            ),
        },
        {
            "series": "mamba_global",
            "path": os.path.join(
                RAW_RESULTS_PATH,
                f"{model}_prefix", "representation_probing", "phonebook",
                "raw_results", "results.json",
            ),
        },
    ]

def plot_representation_probing(
    ax,
    model: MODELS,
    *,
    show_legend: bool = True,
    show_ylabel: bool = True,
    ytick_right: bool = False,
    panel_label: str | None = "(a)",
    marker_layer: int | None = None,
    show_xlabel: bool = True,
):
    for run in _representation_probe_runs(model):
        data = _load_json(run["path"])
        layers = sorted(data["per_layer"].keys(), key=int)
        auc = [data["per_layer"][layer]["auc"] for layer in layers]
        ax.plot(
            [_probe_layer(layer) for layer in layers],
            auc,
            **_plot_kwargs_probe(run["series"]),
        )
    ax.set_ylim(0, 1.02)
    standard.apply_shared_layer_axis(ax, model, show_xlabel=show_xlabel)
    standard.apply_layer_accent(ax, model)
    _add_layer_marker(ax, marker_layer)
    ax.grid(True, alpha=0.3)
    standard.style_axes(ax)
    _configure_yaxis(
        ax, ylabel="Representation Probe AUC",
        show_ylabel=show_ylabel, ytick_right=ytick_right,
    )
    if panel_label is not None:
        standard.panel_label(ax, panel_label)

    if show_legend:
        handles, labels = standard.legend_handles_and_labels(
            standard.LEGEND_SERIES_ORDER
        )
        standard.in_axes_legend(ax, handles, labels, ncol=1)


# ---------------------------------------------------------------------------
# Panel (b): gate-probing probe AUC vs. layer
# ---------------------------------------------------------------------------

def _gate_probe_runs(model: MODELS):
    return [
        {
            "series": "mamba_local",
            "path": os.path.join(
                RAW_RESULTS_PATH,
                f"{model}_prefix", "gate_probing", "phonebook_local",
                "raw_results", "delta_number_last_token_target_results.json",
            ),
        },
        {
            "series": "mamba_global",
            "path": os.path.join(
                RAW_RESULTS_PATH,
                f"{model}_prefix", "gate_probing", "phonebook",
                "raw_results", "delta_number_last_token_results.json",
            ),
        },
    ]


def plot_gate_probing(
    ax,
    model: MODELS,
    *,
    show_legend: bool = True,
    show_ylabel: bool = True,
    ytick_right: bool = False,
    panel_label: str | None = "(b)",
    marker_layer: int | None = None,
    show_xlabel: bool = True,
):
    for run in _gate_probe_runs(model):
        data = _load_json(run["path"])
        layers = sorted(data["per_layer"].keys(), key=int)
        auc = [data["per_layer"][layer]["auc"] for layer in layers]
        ax.plot(
            [_probe_layer(layer) for layer in layers],
            auc,
            **_plot_kwargs_probe(run["series"]),
        )
    ax.set_ylim(0, 1.02)
    standard.apply_shared_layer_axis(ax, model, show_xlabel=show_xlabel)
    standard.apply_layer_accent(ax, model)
    _add_layer_marker(ax, marker_layer)
    ax.grid(True, alpha=0.3)
    standard.style_axes(ax)
    _configure_yaxis(
        ax, ylabel="Gate Probe AUC",
        show_ylabel=show_ylabel, ytick_right=ytick_right,
    )
    if panel_label is not None:
        standard.panel_label(ax, panel_label)

    if show_legend:
        handles, labels = standard.legend_handles_and_labels(
            standard.LEGEND_SERIES_ORDER
        )
        standard.in_axes_legend(ax, handles, labels, ncol=1)

# ---------------------------------------------------------------------------
# Panel (c): patching and scaling (Δ, representation, and scaling)
# ---------------------------------------------------------------------------


def _sweep_rows(data: dict, sweep: str) -> list:
    if sweep == "isolated":
        return data["sweeps"]["isolated"]
    if sweep == "cumulative":
        return standard.cumulative_sweep_rows(data)
    valid = ", ".join(SWEEPS)
    raise ValueError(f"Invalid sweep {sweep!r}. Must be one of: {valid}")


def _sweep_x(layer_range: str, sweep: str) -> int:
    if sweep == "isolated":
        return _layer_index(layer_range)
    return _cumulative_layer_end(layer_range)


def plot_patching_and_scaling(
    ax,
    model: MODELS,
    sweep: str,
    *,
    show_legend: bool = True,
    show_ylabel: bool = True,
    ytick_right: bool = False,
    panel_label: str | None = "(c)",
    marker_layer: int | None = None,
    show_xlabel: bool = False,
):
    patch_components = _patch_components(model)
    baseline_data = _load_json(_baseline_path(model))
    donor_acc = baseline_data["donor_accuracy"]
    recipient_acc = baseline_data["recipient_accuracy"]

    for comp_style in patch_components.values():
        data = _load_json(comp_style["path"])

        rows = _sweep_rows(data, sweep)
        xs = [_sweep_x(row["layer_range"], sweep) for row in rows]
        ys = [row["accuracy"] for row in rows]
        ax.plot(
            xs,
            ys,
            color=comp_style["color"],
            linestyle=comp_style["ls"],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markevery=MARKER_EVERY,
            markeredgewidth=0.3,
        )

    ax.axhline(
        donor_acc, color=LOCAL_BASELINE_COLOR, linestyle=":",
        linewidth=LINE_WIDTH, alpha=0.9, zorder=3,
    )
    ax.axhline(
        recipient_acc, color=GLOBAL_BASELINE_COLOR, linestyle=":",
        linewidth=LINE_WIDTH, alpha=0.9, zorder=3,
    )

    standard.apply_model_acc_ylim(ax, model)
    standard.apply_shared_layer_axis(ax, model, show_xlabel=show_xlabel)
    standard.apply_layer_accent(ax, model)
    _add_layer_marker(ax, marker_layer)
    ax.grid(True, alpha=0.3)
    standard.style_axes(ax)
    _configure_yaxis(
        ax, ylabel="Model acc.",
        show_ylabel=show_ylabel, ytick_right=ytick_right,
    )
    if panel_label is not None:
        standard.panel_label(ax, panel_label)

    if show_legend:
        handles = []
        labels = []
        for comp_style in patch_components.values():
            handles.append(
                Line2D(
                    [0], [0],
                    color=comp_style["color"],
                    linestyle=comp_style["ls"],
                    linewidth=LINE_WIDTH,
                    marker="o",
                    markersize=MARKER_SIZE,
                )
            )
            labels.append(comp_style["label"])
        handles.extend([
            Line2D(
                [0], [0], color=LOCAL_BASELINE_COLOR, linestyle=":",
                linewidth=LINE_WIDTH,
            ),
            Line2D(
                [0], [0], color=GLOBAL_BASELINE_COLOR, linestyle=":",
                linewidth=LINE_WIDTH,
            ),
        ])
        labels.extend([
            "Flagged baseline",
            "Unflagged baseline",
        ])
        standard.in_axes_legend(ax, handles, labels, ncol=1)


def main(model: MODELS, sweep: str = DEFAULT_SWEEP, marker_layer: int | None = None):
    if sweep not in SWEEPS:
        valid = ", ".join(SWEEPS)
        raise ValueError(
            f"Invalid panel (a) sweep {sweep!r}. Must be one of: {valid}"
        )

    output_path = standard.output_path(f"{model}_failure")
    standard.apply_paper_style()

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1, 1, 1],
        hspace=0.10,
        left=0.10,
        right=0.98,
        top=0.95,
        bottom=0.08,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_middle = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top)

    plot_representation_probing(ax_top, model, marker_layer=marker_layer)
    plot_patching_and_scaling(ax_bot, model, sweep, marker_layer=marker_layer)
    plot_gate_probing(ax_middle, model, marker_layer=marker_layer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", pad_inches=0.05, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate failure figure.")
    parser.add_argument(
        "--model",
        default=MODELS.MAMBA_NVIDIA,
        choices=[m.value for m in MODELS],
        help="Model to generate the figure for",
    )
    parser.add_argument(
        "--sweep",
        choices=SWEEPS,
        default=DEFAULT_SWEEP,
        help="Panel (a): patching and scaling sweep type (isolated or cumulative)",
    )
    parser.add_argument(
        "--marker-layer",
        type=int,
        default=None,
        help="Layer to mark with a vertical line (default: no marker)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(parse_model(args.model), sweep=args.sweep, marker_layer=args.marker_layer)
