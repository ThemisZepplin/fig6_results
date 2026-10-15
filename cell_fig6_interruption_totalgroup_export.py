"""Export four-init June 18–20 Fig.6 fields; grouping remains June 14–24 Total.

Raw I/O, BJT daily processing and statistical kernels come unchanged from the
adjacent historical source. This file has no automatic execution block.
"""
from pathlib import Path
import json
import runpy

import numpy as np
import pandas as pd
import xarray as xr

FIG6_INITS = ("2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12")
FIG6_DATES = pd.date_range("2023-06-18", "2023-06-20", freq="D")
FIG6_WINDOW = "2023-06-18 to 2023-06-20"
FIG6_GROUPING_WINDOW = "2023-06-14 to 2023-06-24"
FIG6_ROOT = Path("/data1/huangy/fig6/NC/v12/paper_figures")
FIG6_STEM = "fig6_interruption_0618_0620_totalgroup_fourinit"
SOURCE_FILENAME = "cell_leadtime_total_hotcold_composite_maps.py"


def _fig6_science_namespace(source_path=None):
    if source_path is None:
        folder = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        source_path = folder / SOURCE_FILENAME
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Science source not found: {source_path}. Download both scripts and "
            "use runpy.run_path with their absolute folder; no notebook state is required."
        )
    loaded = runpy.run_path(str(source_path), run_name="fig6_science_source")
    # run_path may return a copy; use the actual function globals, never notebook globals.
    science = loaded["build_daily_s2s_member_fields"].__globals__
    science["ANALYSIS_DATES"] = FIG6_DATES.copy()
    science["ANALYSIS_WINDOW_LABEL"] = FIG6_WINDOW
    if tuple(science["INIT_DATES"]) != FIG6_INITS or science["GROUPING_STAGE"] != "Total":
        raise ValueError("Source must use the four approved initializations and Total grouping")
    return science


def _fig6_member_mean(daily, members, init, variable, domain):
    """Select exactly three BJT days before reducing within each member."""
    dates = pd.DatetimeIndex(daily.bjt_date.values)
    if dates.has_duplicates:
        raise ValueError(f"{init} {variable}: duplicate BJT dates")
    missing = FIG6_DATES.difference(dates)
    if len(missing):
        raise ValueError(f"{init} {variable}: missing BJT dates {list(missing)}")
    selected = daily.sel(bjt_date=FIG6_DATES, member=list(members))
    selected = selected.transpose("member", "bjt_date", "latitude", "longitude")
    expected = 3
    counts = selected.count("bjt_date").transpose("member", "latitude", "longitude")
    lon0, lon1, lat0, lat1 = domain
    mask = ((counts.latitude.values[:, None] >= lat0)
            & (counts.latitude.values[:, None] <= lat1)
            & (counts.longitude.values[None, :] >= lon0)
            & (counts.longitude.values[None, :] <= lon1))
    if not mask.any():
        raise ValueError("No native grid points in the display/significance domain")
    bad = (counts.values < expected) & mask[None, :, :]
    if bad.any():
        raise ValueError(
            f"{init} {variable}: incomplete three-day member fields; "
            f"affected member-gridpoints={int(bad.sum())}"
        )
    valid_counts = counts.values[:, mask]
    qc = {
        "expected_day_count": expected,
        "minimum_valid_day_count": int(valid_counts.min()),
        "maximum_valid_day_count": int(valid_counts.max()),
        "incomplete_member_gridpoints": int(bad.sum()),
    }
    # Same mean operation as historical source, after completeness has passed.
    return selected.mean("bjt_date", skipna=True), qc


def _fig6_reference(science):
    # Both source loaders use the private namespace's three ANALYSIS_DATES.
    reference = science["load_reference_fields"]()
    for variable in ("tp", "z500_abs"):
        field = reference[variable]
        if field.ndim != 2 or not np.isfinite(field.values).all():
            raise ValueError(f"Invalid three-day observation reference: {variable}")
    return reference


def _fig6_product_paths(root):
    root = Path(root)
    return {
        "cache": root / "cache" / f"{FIG6_STEM}_plot_cache.nc",
        "cache_qc": root / "cache" / f"{FIG6_STEM}_cache_qc.csv",
        "tp_significance": root / "significance" / f"{FIG6_STEM}_significance_tp.nc",
        "z500_significance": root / "significance" / f"{FIG6_STEM}_significance_z500.nc",
        "significance_qc": root / "significance" / f"{FIG6_STEM}_significance_qc.csv",
    }


def export_fig6_interruption_cache_and_significance(
    root=FIG6_ROOT, source_path=None, overwrite=False,
):
    """One raw-data pass per init; export means and independent TP/Z500 tests.

    Existing products require explicit overwrite=True. Only the new Fig.6 stem
    is written; historical 11-day and daily products are never replaced.
    """
    paths = _fig6_product_paths(root)
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "New Fig.6 products already exist; redraw from cache, or pass "
            "overwrite=True explicitly for a fresh scientific export:\n" + "\n".join(existing)
        )
    science = _fig6_science_namespace(source_path)
    group_df = science["load_total_hotcold_groups"](science["GROUP_FILE"])
    arrays = {name: [] for name in (
        "tp_hot", "tp_cold", "tp_diff", "z500_abs_hot", "z500_abs_cold", "z500_abs_diff"
    )}
    significance = {"tp": [], "z500_abs": []}
    qc_rows, member_records = [], {}
    for init in FIG6_INITS:
        hot = science["get_group_members"](group_df, init, "Hot17")
        cold = science["get_group_members"](group_df, init, "Cold17")
        science["_fig9_validate_groups"](hot, cold, init)
        if len(set(hot)) != 17 or len(set(cold)) != 17:
            raise ValueError(f"{init}: duplicate members within Hot17 or Cold17")
        member_records[init] = {"Hot17": list(map(int, hot)), "Cold17": list(map(int, cold))}
        fields = science["build_daily_s2s_member_fields"](init)
        if set(map(int, fields["tp"].member.values)) != set(range(51)):
            raise ValueError(f"{init}: TP must contain CF0 and PF1–50")
        if set(map(int, fields["z500_abs"].member.values)) != set(range(51)):
            raise ValueError(f"{init}: Z500 must contain CF0 and PF1–50")
        for variable, units in (("tp", "mm day^-1"), ("z500_abs", "gpm")):
            means, complete = _fig6_member_mean(
                fields[variable], hot + cold, init, variable, science["SIGNIFICANCE_DOMAIN"]
            )
            hot_mean = means.sel(member=hot).mean("member", skipna=True)
            cold_mean = means.sel(member=cold).mean("member", skipna=True)
            difference = hot_mean - cold_mean
            for group, field in (("hot", hot_mean), ("cold", cold_mean), ("diff", difference)):
                arrays[f"{variable}_{group}"].append(
                    science["_cache_map_da_fig9"](
                        field, "s2s", units,
                        f"June 18–20 mean {variable}; Total Tmax {group} group"
                    )
                )
            panel, qc = science["_fig9_welch_panel"](
                means, hot, cold, difference, units, variable, init
            )
            qc.update(complete)
            qc.update(figure="Fig.6", analysis_window=FIG6_WINDOW,
                      grouping_window=FIG6_GROUPING_WINDOW)
            significance[variable].append(panel)
            qc_rows.append(qc)
        print(f"[Fig.6 export] {init}: Total Hot17/Cold17; three-day member means complete")

    # Exact grid matching before concatenation; no coordinate union or interpolation.
    for variable in significance:
        science["_fig9_validate_native_grid_exact"](significance[variable], variable)
    init_coord = pd.DatetimeIndex(FIG6_INITS, name="init")
    fields = {name: xr.concat(items, dim=init_coord, join="exact")
              for name, items in arrays.items()}
    reference = _fig6_reference(science)
    fields["tp_obs_ref"] = science["_cache_map_da_fig9"](
        reference["tp"], "mswep", "mm day^-1", "MSWEP June 18–20 mean daily precipitation")
    fields["z500_abs_obs_ref"] = science["_cache_map_da_fig9"](
        reference["z500_abs"], "era5", "gpm", "ERA5 June 18–20 mean absolute Z500")
    absolute = science["_abs_tp_levels"](
        [reference["tp"], *arrays["tp_hot"], *arrays["tp_cold"]])
    differences, _ = science["_diff_tp_levels"](arrays["tp_diff"])
    fields["tp_abs_levels"] = xr.DataArray(
        absolute, dims="tp_abs_level", attrs={"units": "mm day^-1"})
    fields["tp_diff_levels"] = xr.DataArray(
        differences, dims="tp_diff_level", attrs={"units": "mm day^-1"})
    membership = json.dumps(member_records, sort_keys=True)
    common = {
        "source_figure": "Fig.6: four-init June 18–20 precipitation and circulation",
        "source_script": "cell_fig6_interruption_totalgroup_export.py",
        "science_source_script": SOURCE_FILENAME,
        "science_source_commit": "2f97ecc6bbe6544aa08f6dc36ec3c74b20185058",
        "analysis_window": FIG6_WINDOW, "mean_day_count": 3,
        "target_dates_bjt": ",".join(FIG6_DATES.strftime("%Y-%m-%d")),
        "grouping_stage": "Total", "grouping_window": FIG6_GROUPING_WINDOW,
        "grouping_file": str(science["GROUP_FILE"]), "group_members_by_init": membership,
        "sign_convention": "Hot17_minus_Cold17 = Hot17 - Cold17",
        "z500_type": "absolute", "z500_units": "gpm", "tp_units": "mm day^-1",
        "tp_aggregation": "arithmetic mean of three BJT daily accumulated amounts",
        "z500_diff_type": "Hot17_minus_Cold17 difference of absolute Z500",
        "map_extent": str(science["MAP_EXTENT"]), "nchn_box": str(science["NCHN_BOX"]),
        "native_grid_policy": "separate s2s/mswep/era5 coordinates; no regridding",
    }
    cache = xr.Dataset(fields, attrs=common)
    science["_validate_fig9_s2s_cache_density"](cache)
    sig_datasets = {}
    group_rows = [f"{i}:Hot17={member_records[i]['Hot17']};Cold17={member_records[i]['Cold17']}"
                  for i in FIG6_INITS]
    for variable, units in (("tp", "mm day^-1"), ("z500_abs", "gpm")):
        ds = xr.concat(significance[variable], dim="init", join="exact")
        ds = ds.assign_coords(init=init_coord)
        ds.attrs.update(science["_fig9_significance_attrs"](units, group_rows))
        ds.attrs.update(common)
        ds.attrs.update(
            units=units, difference_direction="Hot17 minus Cold17",
            expected_target_period_day_count=3,
            target_period_completeness_required="3 valid BJT days for each selected member/gridpoint in domain",
            main_figure_criterion="uncorrected local two-sided Welch p < 0.05",
        )
        expected = cache[f"{variable}_diff"]
        if not np.allclose(ds.hot_minus_cold, expected, rtol=1e-7, atol=1e-6, equal_nan=True):
            raise ValueError(f"{variable}: significance/cache differences do not match")
        sig_datasets[variable] = ds
    cache_rows = science["_cache_qc_rows_fig9"](cache, str(paths["cache"]))
    for row in cache_rows:
        row.update(figure="Fig.6", analysis_window=FIG6_WINDOW,
                   grouping_window=FIG6_GROUPING_WINDOW, mean_day_count=3)
    # All calculations/checks finish before any scientific product is written.
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_netcdf(paths["cache"])
    sig_datasets["tp"].to_netcdf(paths["tp_significance"])
    sig_datasets["z500_abs"].to_netcdf(paths["z500_significance"])
    pd.DataFrame(cache_rows).to_csv(paths["cache_qc"], index=False)
    pd.DataFrame(qc_rows).to_csv(paths["significance_qc"], index=False)
    for label, path in paths.items():
        print(f"[Fig.6 {label}] {path}")
    return {label: str(path) for label, path in paths.items()}
