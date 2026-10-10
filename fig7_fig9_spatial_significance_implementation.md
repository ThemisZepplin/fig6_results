# Fig.7/Fig.9 spatial-significance implementation inventory

## Scope

This inventory documents the significance-only extension. It does not report scientific results: the configured raw S2S files are not available in this repository, so the NetCDF, QC CSV, and runtime audit must be generated and reviewed in the HPC/Jupyter environment.

## Source inventory

### Fig.7

- Source: `cell_fig7_stageII_daily_synoptic_paper.py`.
- Main initialization: `2023-06-12` (`MAIN_INIT_DATE`).
- Dates: `2023-06-18`, `2023-06-19`, and `2023-06-20` (`TARGET_DATES`).
- Group source: `/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv`; stage `Total`.
- Member IDs are read by `load_total_hotcold_groups_fig7()` and `get_hotcold_members_fig7()`; both groups are required to contain exactly 17 distinct members. No new ranking is performed.
- Member-level daily fields are returned by `build_daily_realtime_member_fields_fig7()`. Their dimensions are `member`, `bjt_date`, `latitude`, and `longitude`; TP is in mm day-1 and Z500 is in gpm.
- `build_group_composites_fig7()` computes each group mean over `member`, then defines `Hot17_minus_Cold17` as Hot17 minus Cold17.
- `export_fig7_spatial_significance()` tests each date independently. Member is the sole sample dimension; dates are neither pooled nor treated as additional samples.
- Six independent BH-FDR families are used: TP and Z500 separately for each of the three dates. Each family is restricted a priori to 90–135 E, 25–55 N on the native S2S grid.

### Fig.9

- Source: `cell_leadtime_total_hotcold_composite_maps.py`.
- Initializations: `2023-06-01`, `2023-06-05`, `2023-06-08`, and `2023-06-12` (`INIT_DATES`).
- Analysis window: `2023-06-14` through `2023-06-24` BJT (`ANALYSIS_DATES`).
- Group source: `/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv`; stage `Total`. Groups are loaded independently for each initialization, with exactly 17 distinct members required in each group and no new ranking.
- `build_daily_s2s_member_fields()` returns daily member-level TP and absolute Z500 fields with dimensions `member`, `bjt_date`, `latitude`, and `longitude` (mm day-1 and gpm).
- `_period_mean()` first averages `bjt_date` separately within each member. `build_total_composites_for_init()` subsequently averages those member-level period means over the selected Hot17 or Cold17 members. It does not form an ensemble mean before the period mean.
- `export_fig9_spatial_significance()` follows the same order: member daily field, member-level Total-period mean, then a 17-versus-17 gridpoint test. Initializations and days are never pooled as samples.
- Eight independent BH-FDR families are used: TP and absolute-Z500 difference separately for each initialization. Each family is restricted a priori to 90–135 E, 25–55 N on the native S2S grid.

## Statistical implementation

- Test: two-sided Welch independent-samples t test (`scipy.stats.ttest_ind`, `equal_var=False`, `nan_policy="omit"`). The explicit `alternative="two-sided"` argument is used when supported; the runtime audit records whether the fallback to SciPy's two-sided default was needed.
- Null hypothesis: the gridpoint Hot17 and Cold17 member means are equal.
- Multiple testing: `statsmodels.stats.multitest.multipletests(..., method="fdr_bh", alpha=0.05)` independently within each panel × variable family.
- Stored adjusted quantity: `p_fdr_bh`, explicitly a Benjamini–Hochberg-adjusted p-value rather than a Storey q-value.
- Saved diagnostics: `hot_mean`, `cold_mean`, `hot_minus_cold`, `t_statistic`, `p_value`, `p_fdr_bh`, local and FDR significance masks, and finite member counts. Welch degrees of freedom are also saved when supplied by SciPy.
- Consistency guard: the member-derived Hot17-minus-Cold17 field must match the original plotted composite to numerical tolerance before output is written.
- Grid handling: native S2S coordinates are retained. No interpolation, regridding, mask expansion, or morphology is performed. Padding outside the final 90–135 E, 25–55 N display domain is excluded from every FDR family.

## Expected runtime outputs

- `/data1/huangy/fig6/NC/v12/paper_figures/significance/fig7_spatial_significance_tp.nc`
- `/data1/huangy/fig6/NC/v12/paper_figures/significance/fig7_spatial_significance_z500.nc`
- `/data1/huangy/fig6/NC/v12/paper_figures/significance/fig9_spatial_significance_tp.nc`
- `/data1/huangy/fig6/NC/v12/paper_figures/significance/fig9_spatial_significance_z500.nc`
- `/data1/huangy/fig6/NC/v12/paper_figures/significance/fig7_spatial_significance_qc.csv`
- `/data1/huangy/fig6/NC/v12/paper_figures/significance/fig9_spatial_significance_qc.csv`
- `/data1/huangy/fig6/NC/v12/paper_figures/significance/fig7_fig9_spatial_significance_audit.md`
- `/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/fig7_daily_synoptic_evolution_from_cache_sigfdr.png` and `.pdf`
- `/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/fig9_leadtime_tp_z500_composites_from_cache_sigfdr.png` and `.pdf`

## Plotting contract

`paper_fig7_from_cache.py` and `paper_fig9_from_cache.py` read only the existing plot-ready cache and the two corresponding significance NetCDF files. They do not import SciPy/statsmodels, read grouping tables or raw forecast data, or recompute tests/FDR. Hatching is applied only to FDR-significant TP differences in Hot17-minus-Cold17 panels. Existing Z500 difference contours remain visible everywhere; the same contour levels are redrawn more strongly only where the independently calculated Z500 FDR mask is true.

## Caption supplements

**Fig.7.** Hatching denotes grid points where the Hot17−Cold17 precipitation difference is significant based on a two-sided Welch’s t test after Benjamini–Hochberg false-discovery-rate adjustment (FDR-adjusted p < 0.05). Thickened Z500 contours indicate portions of the Hot17−Cold17 geopotential-height difference satisfying the same variable-specific criterion. Precipitation and Z500 were tested separately using the 17 members in each predefined group. Tests were conducted separately for each day.

**Fig.9.** Hatching denotes grid points where the Hot17−Cold17 precipitation difference is significant based on a two-sided Welch’s t test after Benjamini–Hochberg false-discovery-rate adjustment (FDR-adjusted p < 0.05). Thickened Z500 contours indicate portions of the Hot17−Cold17 geopotential-height difference satisfying the same variable-specific criterion. Precipitation and Z500 were tested separately using the 17 members in each predefined group. Tests were conducted independently for each initialization using member-level Total-period means.

## Interpretation boundary

The tests diagnose conditional spatial differences between predefined ensemble-member groups selected using the established Total-period Tmax ranking. They do not establish causality, independently validate that ranking, test the significance of the 2023 event itself, or turn members from one forecast system into independent observational population samples. BH-FDR addresses multiplicity among gridpoint tests within each declared family and should not be described as stronger “global field significance.”
