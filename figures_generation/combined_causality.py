"""Stacked causality figure: NVIDIA (top) and Codestral (bottom)."""

import matplotlib.pyplot as plt

import causality
import standard
from standard import MODELS

FIG_WIDTH = causality.FIG_WIDTH
FIG_HEIGHT = causality.FIG_HEIGHT


def main():
    output_path = standard.output_path("combined_causality")
    standard.apply_paper_style()

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        2, 1,
        hspace=0.25,
        left=0.10,
        right=0.99,
        top=0.90,
        bottom=0.22,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])

    causality.plot(
        ax_top, MODELS.MAMBA_NVIDIA,
        show_xticks=False, show_legend=False,
        show_section_labels=True,
    )
    causality.plot(
        ax_bot, MODELS.MAMBA_CODESTRAL,
        show_xticks=True, show_legend=True,
    )

    ax_top.set_title(
        MODELS.MAMBA_NVIDIA.display_name, fontsize=standard.MODEL_TITLE_SIZE,
        fontweight="bold", loc="center", pad=3,
    )
    ax_bot.set_title(
        MODELS.MAMBA_CODESTRAL.display_name, fontsize=standard.MODEL_TITLE_SIZE,
        fontweight="bold", loc="center", pad=3,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", pad_inches=0.05, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
