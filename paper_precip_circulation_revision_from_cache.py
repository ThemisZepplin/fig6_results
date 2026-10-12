"""Unambiguous JGR figure entries; resolve sibling scripts from this file."""
from pathlib import Path
import runpy


def _entry(filename, function, **kwargs):
    namespace = runpy.run_path(str(Path(__file__).resolve().parent / filename))
    return namespace[function](**kwargs)


def plot_fig6_from_cache(**kwargs):
    return _entry("paper_fig7_from_cache.py", "plot_fig6_from_cache", **kwargs)


def plot_figS6_from_cache(**kwargs):
    return _entry("paper_fig7_from_cache.py", "plot_figS6_from_cache", **kwargs)


def plot_fig7_from_cache(**kwargs):
    return _entry("paper_fig8_from_cache.py", "plot_fig7_hovmoller_from_cache", **kwargs)


def plot_fig9_from_cache(**kwargs):
    return _entry("paper_fig9_from_cache.py", "plot_fig9_from_cache", **kwargs)
