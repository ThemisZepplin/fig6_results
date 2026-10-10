"""Minimal Fig.7–Fig.9 redraw example from plot-ready caches."""

from paper_fig7_from_cache import plot_fig7_from_cache
from paper_fig8_from_cache import plot_fig8_from_cache
from paper_fig9_from_cache import plot_fig9_from_cache

OUT_DIR = "/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache"

plot_fig7_from_cache(
    "/data1/huangy/fig6/NC/v12/paper_figures/cache/fig7_daily_synoptic_plot_cache.nc",
    OUT_DIR,
)
plot_fig8_from_cache(
    "/data1/huangy/fig6/NC/v12/paper_figures/cache/fig8_hovmoller_plot_cache.nc",
    OUT_DIR,
)
plot_fig9_from_cache(
    "/data1/huangy/fig6/NC/v12/paper_figures/cache/fig9_leadtime_tp_z500_plot_cache.nc",
    OUT_DIR,
)
