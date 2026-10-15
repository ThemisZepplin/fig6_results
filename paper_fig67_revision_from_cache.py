"""Explicit entries for the October 2026 Fig.6/Fig.7 revision."""
from pathlib import Path
import runpy

ROOT = Path("/data1/huangy/fig6/NC/v12/paper_figures")


def _entry(filename, function, **kwargs):
    folder = Path(__file__).resolve().parent
    path = folder / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required sibling script missing: {path}")
    namespace = runpy.run_path(str(path), run_name="fig67_renderer")
    return namespace[function](**kwargs)


def plot_fig6_from_cache(**kwargs):
    """New four-init June 18–20 main-text map."""
    return _entry("paper_fig6_interruption_from_cache.py", "plot_fig6_from_cache", **kwargs)


def plot_fig7_from_cache(**kwargs):
    """Existing seven panels, revised headings and M-D dates."""
    return _entry("paper_fig8_from_cache.py", "plot_fig7_hovmoller_from_cache", **kwargs)


def plot_daily_si_candidate(initialization, **kwargs):
    """Preserved June 8/12 daily maps; final SI numbering remains undecided."""
    if initialization not in ("2023-06-08", "2023-06-12"):
        raise ValueError("Daily SI candidates support June 8 and June 12 only")
    kwargs.setdefault("out_dir", str(ROOT / "final_from_cache" / "si_daily_candidates"))
    function = "plot_figS6_from_cache" if initialization == "2023-06-08" else "plot_fig6_from_cache"
    return _entry("paper_fig7_from_cache.py", function, **kwargs)


def plot_total_mean_archive_from_cache(**kwargs):
    """Former manuscript Fig.8, historically called Fig.9 in code."""
    kwargs.setdefault("out_dir", str(ROOT / "final_from_cache" / "archive_total_0614_0624"))
    return _entry("paper_fig9_from_cache.py", "plot_fig9_from_cache", **kwargs)
