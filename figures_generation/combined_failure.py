"""Side-by-side failure figure: NVIDIA (left) and Codestral (right)."""

import matplotlib.pyplot as plt

import failure
import standard
from standard import MODELS

FIG_WIDTH = failure.FIG_WIDTH
FIG_HEIGHT = failure.FIG_HEIGHT


def main(sweep: str = failure.DEFAULT_SWEEP):
    output_path = standard.output_path("combined_failure")
    standard.apply_paper_style()

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1, 1, 1],
        hspace=0.10,
        wspace=0.04,
        left=0.10,
        right=0.92,
        top=0.90,
        bottom=0.08,
    )
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_cr = fig.add_subplot(gs[1, 0], sharex=ax_tl)
    ax_cl = fig.add_subplot(gs[1, 1], sharex=ax_tr)
    ax_bl = fig.add_subplot(gs[2, 0], sharex=ax_tl)
    ax_br = fig.add_subplot(gs[2, 1], sharex=ax_tr)
    
    failure.plot_representation_probing(
        ax_tl, MODELS.MAMBA_NVIDIA,
        show_legend=False, panel_label="(a)",
    )
    failure.plot_representation_probing(
        ax_tr, MODELS.MAMBA_CODESTRAL,
        show_legend=True, show_ylabel=False, ytick_right=True,
        panel_label=None,
    )

    failure.plot_gate_probing(
        ax_cr, MODELS.MAMBA_NVIDIA,
        show_legend=False, panel_label="(b)",
    )
    failure.plot_gate_probing(
        ax_cl, MODELS.MAMBA_CODESTRAL,
        show_legend=True, show_ylabel=False, ytick_right=True,
        panel_label=None,
    )

    failure.plot_patching_and_scaling(
        ax_bl, MODELS.MAMBA_NVIDIA, sweep,
        show_legend=False, panel_label="(c)",
    )
    failure.plot_patching_and_scaling(
        ax_br, MODELS.MAMBA_CODESTRAL, sweep,
        show_legend=True, show_ylabel=False, ytick_right=True, panel_label=None,
    )

    ax_tl.set_title(MODELS.MAMBA_NVIDIA.display_name, fontsize=standard.MODEL_TITLE_SIZE, fontweight="bold", pad=3)
    ax_tr.set_title(MODELS.MAMBA_CODESTRAL.display_name, fontsize=standard.MODEL_TITLE_SIZE, fontweight="bold", pad=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", pad_inches=0.05, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
