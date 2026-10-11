"""Repo-relative output roots, overridable via environment variables.

Layout:

  $RAW_RESULTS_PATH/<model>_<task>/<experiment>/<dataset>/raw_results/
  $PLOTS_PATH/<model>_<task>/<experiment>/<dataset>/plots/
  $FIGURES_PATH/<name>.pdf

Resolution:

  OUTPUT_DIR         default <repo>/output_dir
  RAW_RESULTS_PATH   default OUTPUT_DIR
  PLOTS_PATH         default OUTPUT_DIR
  FIGURES_PATH       default OUTPUT_DIR/figures
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str) -> Path | None:
	value = os.environ.get(name)
	return Path(value).expanduser() if value else None


OUTPUT_DIR = _env_path("OUTPUT_DIR") or (REPO_ROOT / "output_dir")
RAW_RESULTS_PATH = _env_path("RAW_RESULTS_PATH") or OUTPUT_DIR
PLOTS_PATH = _env_path("PLOTS_PATH") or OUTPUT_DIR
FIGURES_PATH = _env_path("FIGURES_PATH") or (OUTPUT_DIR / "figures")


def experiment_raw_results_dir(experiment_subdir: str, *extra: str) -> str:
	path = RAW_RESULTS_PATH.joinpath(experiment_subdir, "raw_results", *extra)
	path.mkdir(parents=True, exist_ok=True)
	return str(path) + os.sep


def experiment_plots_dir(experiment_subdir: str, *extra: str) -> str:
	path = PLOTS_PATH.joinpath(experiment_subdir, "plots", *extra)
	path.mkdir(parents=True, exist_ok=True)
	return str(path) + os.sep


def figure_pdf_path(name: str) -> Path:
	FIGURES_PATH.mkdir(parents=True, exist_ok=True)
	return FIGURES_PATH / f"{name}.pdf"
