# Methods audit: spatial propagation and Fig.7–Fig.9 cache/from-cache workflow

> **Update notice:** This audit predates the member-level Welch/BH-FDR significance extension now documented in `fig7_fig9_spatial_significance_implementation.md`. Statements below that spatial significance was “not found” describe the pre-extension snapshot and are superseded only for the new Fig.7/Fig.9 Hot17−Cold17 significance workflow; all other audit findings remain unchanged.

Scope: this audit uses only files present in the current repository window. It does **not** infer methods from unavailable notebooks, HPC jobs, prior manuscript text, or domain convention. No code was executed except text search/inspection commands, and no science-processing workflow was run.

## 0. Files scanned and evidence basis

Primary source files scanned:

- `cell_fig7_stageII_daily_synoptic_paper.py`: Fig.7 daily synoptic maps, Total-stage Hot17/Cold17 grouping for 2023-06-08 and 2023-06-12, S2S daily fields, vectors, cache export.
- `cell_hovmoller_tp_z500_obs_s2s_stageII.py`: Fig.8 Hovmöller diagnostics, Total-stage grouping for 2023-06-08 and 2023-06-12, cache export.
- `cell_leadtime_total_hotcold_composite_maps.py`: Fig.9 lead-time Total-stage composites for four initializations, absolute Z500, cache export and QC.
- `cell_y_era5_z500_uv850_tp_daily_evolution.py`: ERA5/MSWEP daily evolution and ERA5 Z500 anomaly against 2003–2022 climatology.
- `cell_z_s2s_z500_uv850_tp_qflux_daily_evolution.py`: S2S daily evolution, realtime-minus-hindcast Z500 anomaly, q-flux diagnostic.
- `cell_x_era5_mswep_daily_evolution.py`: ERA5/MSWEP precipitation, q-flux and MFC diagnostic.
- `cell_zz_s2s_stageII_hotcold_daily_circulation.py`: S2S Stage-II hot/middle/cold daily circulation diagnostic.
- `paper_fig7_from_cache.py`, `paper_fig8_from_cache.py`, `paper_fig9_from_cache.py`: style-only plotting from cache; not science-processing definitions.

Code searches used during audit included `rg` queries for terms such as `anomaly`, `climat`, `significance`, `ttest`, `RMSE`, `Bias`, `MLR`, `lmg`, `johnson`, `LOFO`, `partial`, `resid`, `KDE`, `Hot17`, `Cold17`, `BJT`, `apportion`, and `member`.

---

## 1. Unified method summary table

| Method | Code location | Input | Sample dimension | Formula | Output unit | Main purpose | Interpretation boundary | Needs citation |
|---|---|---|---|---|---|---|---|---|
| Data source and variable loading | Fig.7 config and loaders; Fig.8/Fig.9 configs | ERA5 TP/Z500/UV850, MSWEP precipitation, ECMWF S2S CF/PF GRIB | time/date, member, lat/lon | variable matching by candidate names and metadata | variable-dependent | Define reproducible data inputs for maps/Hovmöllers/composites | Does not document external data product versions beyond paths | Dataset citations needed |
| Unit conversion | Fig.7 ERA5/S2S loaders, Fig.8/Fig.9 loaders | TP, Z500 | all grid/time/member dims | TP meters ×1000; geopotential / 9.80665 when units/magnitude indicate geopotential | mm day^-1, gpm | Put precipitation and Z500 into plotted units | Unit heuristics depend on attrs and magnitude checks | Data-processing citation not needed; data-product docs needed |
| BJT daily aggregation | Fig.7 `_to_bjt_daily_fig7`, S2S `_build_bjt_valid_coord_fig7`; Fig.9 analogous functions | hourly/3-hourly/realtime step fields | time or step | UTC/valid time + 8 h, then daily sum or mean | variable-dependent | Define local-day daily fields | BJT day is computational definition, not independent physical filtering | No method citation needed |
| S2S accumulated TP apportionment | Fig.7 `_apportion_accum_to_calendar_days_fig7`; Fig.9 `_apportion_accum_to_calendar_days` | accumulated S2S TP on forecast steps | step, member, lat/lon | step increments are allocated to BJT calendar days by interval overlap fraction | mm day^-1 | Prevent loss/misalignment of accumulated precipitation across BJT days | Assumes increments are uniform across the interval when split by calendar-day fractions | ECMWF accumulation convention documentation recommended |
| Spatial crop and native-grid plotting/cache | Fig.7/Fig.9 `DATA_EXTENT`, `MAP_EXTENT`, cache helpers; Fig.8 `LAT_BAND`, `LON_RANGE` | gridded fields | lat/lon or longitude | `.sel()` over fixed extent; cache dims source-specific | unchanged | Keep maps/Hovmöllers in analysis domain without regridding | No interpolation is performed; different sources retain native grids in cache | No citation needed |
| Cos(latitude) area mean | Cell Y `_area_mean`; Fig.8 `_latband_hov` | gridded fields | lat/lon | weighted mean with weights cos(latitude) | variable-dependent | Approximate area weighting for rectangular lat-lon grids | Applies only where function is called; not all diagnostics use area mean | Standard geophysical practice; optional citation |
| ERA5 Z500 climatology/anomaly | Cell Y `load_z500_clim_daily_2003_2022`, run logic | ERA5 Z500 June files, years 2003–2022 | clim_year, time, lat/lon | anomaly = 2023 daily Z500 − mean over clim_year for same month-day | gpm | Compare 2023 circulation with historical June daily climatology | Current Fig.7–Fig.9 final caches use absolute Z500 except Cell Y anomaly diagnostic | ERA5/data citation; anomaly method can be described |
| S2S realtime-minus-hindcast Z500 anomaly | Cell Z `build_daily_s2s_fields_for_init` | realtime and hindcast S2S Z500 | member/samples, bjt_date, lat/lon | z500_anom = mean_member(realtime Z500) − sample_mean(hindcast Z500) | gpm | Diagnose forecast anomaly relative to reforecast baseline | Not used in current Fig.9 lead-time cache, which uses absolute Z500 | S2S/reforecast citation needed |
| Total-stage Hot17/Cold17 grouping | Fig.7 `load_total_hotcold_groups_fig7`; Fig.9 `load_total_hotcold_groups` | grouping CSV | init_date, stage, group, member | select stage normalized to Total, then parse 17 members for each group | member IDs | Select high/low Tmax-error-related member groups | Grouping variable itself is not recomputed in these scripts; design must be documented from grouping-generation code elsewhere | Needs citation/design explanation if based on Tmax ranking |
| CF/PF merge to 51 members | Fig.7 `_concat_cf_pf_fig7`; Fig.9 `_concat_cf_pf` | control and perturbed streams | member | normalize member dim and concatenate with member IDs 0..50 | member index | Build S2S ensemble before group selection | Assumes CF/PF files are correctly paired and member IDs are offset consistently | ECMWF ensemble documentation recommended |
| Hot17/Cold17 composites | Fig.7/Fig.9 composite builders; Fig.8 Hovmöller group builders | member fields and group member lists | member, bjt_date/time, lat/lon/lon | group mean over selected members | variable-dependent | Compare physical fields of warm/cold member groups | Group means are conditional composites, not causal effects | Composite-analysis citation optional |
| Hot17 − Cold17 difference | Fig.7/Fig.9 builders, cache attrs | Hot17 and Cold17 composites | same as composite field | Hot17_minus_Cold17 = Hot17 − Cold17 | variable-dependent | Show contrast between two member groups | Sign depends on group definition; not a significance-tested causal effect here | No separate citation |
| Hovmöller propagation diagnostic | Fig.8 `_latband_hov` and cache | TP/Z500 over lat band | bjt_date, longitude | cos-lat weighted mean over 35–44N and longitude series | mm day^-1, gpm | Diagnose time-longitude propagation | Latitude-band mean compresses meridional structure | Hovmöller reference optional |
| Daily synoptic composites | Fig.7 plot/build functions | daily TP/Z500/UV fields | date, lat/lon | daily maps for 18–20 Jun | variable-dependent | Show synoptic evolution during key dates | Descriptive; does not test significance | No method citation needed |
| Lead-time Total composites | Fig.9 `build_total_composites_for_init` | period-mean TP/Z500 for four init dates | init, member, lat/lon | period mean over 14–24 Jun, then group mean | mm day^-1, gpm | Assess initialization dependence of Total-stage composites | Multi-init comparison is descriptive unless separately tested | No method citation needed |
| Cache/QC diagnostics | cache export functions | plot-ready fields | variable dims | min/max/mean/nan_count/finite_fraction; TP max location in Fig.9 | variable-dependent | Reproducibility and sanity checking | QC is not a scientific estimator unless reported as such | No citation needed |
| MLR / LMG / Johnson / LOFO / partial fitted component | search result | not found in current window | not applicable | not implemented here | not applicable |本窗口未使用 | Must be audited in another window before Methods text | Original method citations required elsewhere |
| Bias/RMSE/spread/quantiles/KDE/rank | search result | not found in current window | not applicable | not implemented here | not applicable |本窗口未使用 | Must be audited in error-metrics/history window | Citations/design text needed elsewhere |
| p<0.05 spatial significance | search result | not found in current window | not applicable | not implemented here | not applicable |本窗口未使用 | Do not claim significance for these plotted differences | Test-method citation needed if used elsewhere |

---

## 2. Method cards

## 2.1 Data sources and variable loading

### 1. Scientific purpose
To define the gridded observed/reanalysis and S2S forecast fields used to describe precipitation and circulation evolution during the June 2023 North China heatwave period.

### 2. Code evidence
- Fig.7 constants define target dates, domains, data paths for ERA5 TP/Z500/UVQ, MSWEP, S2S CF/PF paths, and Total-stage grouping CSV. `TARGET_DATES`, `MAIN_INIT_DATE`, `APPENDIX_INIT_DATE`, `MAP_EXTENT`, `DATA_EXTENT`, `NCHN_BOX`, `BJT_OFFSET`, `GROUP_FILE`, and S2S paths are defined in `cell_fig7_stageII_daily_synoptic_paper.py:30-90`.
- Fig.9 constants define `INIT_DATES`, `ANALYSIS_DATES`, `GROUPING_STAGE`, `MAP_EXTENT`, `DATA_EXTENT`, `NCHN_BOX`, and `GROUP_FILE` in `cell_leadtime_total_hotcold_composite_maps.py:24-49`.
- Variable detection uses candidate names such as `tp`, `total_precipitation`, `z`, `geopotential`, `gh`, and `hgt` in Fig.7 and Fig.9 loaders.

### 3. Current code's exact definition
- Fig.7 uses `TARGET_DATES = 2023-06-18, 2023-06-19, 2023-06-20`, main init `2023-06-12`, appendix init `2023-06-08`, and map extent 90–135E, 25–55N while reading fields over a broader data extent 88–137E, 23–57N.
- Fig.8 uses `WINDOWS["0614_0624"] = 2023-06-14 to 2023-06-24`, init dates `2023-06-08` and `2023-06-12`, latitude band 35–44N, longitude range 90–135E, and data extent 88–137E, 33–46N.
- Fig.9 uses four initialization dates: 2023-06-01, 2023-06-05, 2023-06-08, and 2023-06-12; analysis window is 2023-06-14 to 2023-06-24; grouping stage is `Total`.
- Data are read from hard-coded local paths. The code does not expose external dataset version metadata beyond these paths.

### 4. Mathematical formula
No statistical formula is applied at the data-loading stage. For a variable field \(A\), code maps an input file variable name to an internal field:

\[
A = \operatorname{guess\_var}(D, C),
\]

where \(D\) is an opened dataset and \(C\) is a list of candidate variable names.

### 5. Method principle
General principle: robust loading maps heterogeneous file variable names to a consistent internal variable. Current implementation uses candidate names, GRIB metadata when available, and fallback logic. It is a reproducibility device, not a scientific estimator.

### 6. Output and interpretation
Outputs are xarray `DataArray` fields for TP, Z500, U/V850, humidity/wind-derived fluxes, and S2S member fields. Their units are set or converted later. Output paths and variable names should be reported for reproducibility, but file paths alone are not complete dataset citations.

### 7. Difference from adjacent methods
Data loading differs from preprocessing: loading finds variables and opens files; preprocessing changes time, units, region, or aggregation.

### 8. Assumptions and limitations
- The correct file is assumed to exist at the configured path.
- Variable guessing is heuristic and depends on candidate names and metadata.
- The code does not record formal data product versions or retrieval dates.

### 9. Methods-ready text
中文：本研究使用 ERA5、MSWEP 和 ECMWF S2S 集合预报资料构建降水和 500 hPa 位势高度诊断场。各脚本通过候选变量名和 GRIB/NetCDF 元数据识别降水、Z500、850 hPa 风场等变量，并在后续步骤中统一时间、空间范围和单位。

English: ERA5, MSWEP, and ECMWF S2S ensemble fields were loaded using variable-name and metadata-based matching. The retrieved fields include precipitation, 500-hPa geopotential height, and, where required, 850-hPa wind components. Subsequent preprocessing harmonized the time convention, spatial domain, and units before compositing or plotting.

### 10. Still needs confirmation/literature
- Code cannot confirm formal dataset product versions or retrieval dates.
- Author should confirm exact ERA5/MSWEP/S2S data products and resolution for Methods.
- Suggested search terms: “ERA5 reanalysis Hersbach et al.”, “MSWEP precipitation dataset”, “ECMWF S2S database”.

---

## 2.2 Unit conversion for precipitation and Z500

### 1. Scientific purpose
To ensure that precipitation is plotted/analyzed in mm day^-1 and 500-hPa geopotential height is in geopotential meters (gpm), so observed and forecast fields are comparable.

### 2. Code evidence
- ERA5 TP in meters is multiplied by 1000 before daily summation in `load_era5_tp_fig7`; output units set to `mm day^-1` (`cell_fig7_stageII_daily_synoptic_paper.py:279-291`).
- ERA5 Z500 is divided by `G0 = 9.80665` if units or magnitude indicate geopotential, then units are set to `gpm` (`cell_fig7_stageII_daily_synoptic_paper.py:294-309`).
- Fig.9 uses `_to_gpm_if_needed` for S2S Z500 before BJT daily mean and stores `z500_abs` in gpm (`cell_leadtime_total_hotcold_composite_maps.py:505-519`).
- Cache variables record units such as `mm day^-1` and `gpm` in Fig.9 export (`cell_leadtime_total_hotcold_composite_maps.py:1001-1009`).

### 3. Current code's exact definition
- Precipitation: if source units indicate meters, values are multiplied by 1000. For S2S accumulated TP, unit conversion is applied after accumulation increments are apportioned to BJT calendar days.
- Z500: if variable units or magnitude suggest geopotential (m^2 s^-2), values are divided by 9.80665. If already height-like, the field remains unchanged and is labeled gpm.

### 4. Mathematical formula
For precipitation in meters:

\[
P_{\mathrm{mm}} = 1000 P_{\mathrm{m}}.
\]

For geopotential \(\Phi\):

\[
Z_{500} = \frac{\Phi_{500}}{g_0}, \quad g_0 = 9.80665\,\mathrm{m\,s^{-2}}.
\]

### 5. Method principle
General principle: accumulated precipitation depth and geopotential height must be placed in common physical units. Current code uses units strings and magnitude heuristics to decide whether conversion is needed.

### 6. Output and interpretation
- TP: mm day^-1.
- Z500: gpm.
Values should not be compared before conversion.

### 7. Difference from adjacent methods
Unit conversion is distinct from anomaly computation: conversion changes physical units; anomaly subtracts a climatological baseline.

### 8. Assumptions and limitations
- Unit metadata must be accurate or magnitude heuristic must correctly detect geopotential.
- Repeated conversion would distort Z500; code guards partly by checking units/magnitude.

### 9. Methods-ready text
中文：降水统一转换为 mm day^-1。若降水变量以米为单位，则乘以 1000。500 hPa 位势若以 m^2 s^-2 表示，则除以标准重力加速度 9.80665 m s^-2 转换为 geopotential meters。

English: Precipitation was expressed in mm day^-1. Fields stored as meters were multiplied by 1000. Geopotential at 500 hPa was converted to geopotential meters by dividing by 9.80665 m s^-2 when the metadata or value range indicated geopotential units.

### 10. Still needs confirmation/literature
- Confirm raw S2S TP accumulation units from ECMWF file documentation.
- Dataset documentation should be cited for native variable units.

---

## 2.3 Beijing Time (BJT) daily aggregation

### 1. Scientific purpose
To define daily fields using local Beijing Time calendar days, matching the event timing over North China.

### 2. Code evidence
- Fig.7 defines `BJT_OFFSET = pd.Timedelta(hours=8)` (`cell_fig7_stageII_daily_synoptic_paper.py:38`).
- General daily conversion adds 8 hours to `time`, then resamples to 1-day bins using sum or mean (`cell_fig7_stageII_daily_synoptic_paper.py:215-223`).
- S2S valid times are built from `init_date + step + BJT_OFFSET` or shifted valid_time; daily means are created by resampling to `1D` and renaming to `bjt_date` (`cell_fig7_stageII_daily_synoptic_paper.py:577-591`).
- Fig.9 uses analogous BJT valid-time construction and daily mean/accumulation functions (`cell_leadtime_total_hotcold_composite_maps.py:437-488`).

### 3. Current code's exact definition
- For non-accumulated fields such as Z500, U850, V850: shift time/valid time by +8 hours and daily mean.
- For precipitation: shift time and daily sum for observed/MSWEP fields; for S2S accumulated TP, use step-difference apportionment before summing by BJT date.

### 4. Mathematical formula
For instantaneous or mean-rate field \(A(\tau)\) with UTC time \(\tau\):

\[
\tau_{\mathrm{BJT}} = \tau + 8\,\mathrm{h}.
\]

Daily mean:

\[
\bar{A}_{d} = \frac{1}{n_d}\sum_{\tau_{\mathrm{BJT}}\in d} A(\tau).
\]

Daily sum for precipitation samples:

\[
P_d = \sum_{\tau_{\mathrm{BJT}}\in d} P(\tau).
\]

### 5. Method principle
General principle: shift timestamps to local time before daily aggregation. Current implementation applies a fixed UTC+8 offset with no daylight-saving adjustment.

### 6. Output and interpretation
Output daily coordinates are named `bjt_date` or later `date/time` in cache. These fields represent BJT calendar days.

### 7. Difference from adjacent methods
Daily aggregation precedes group compositing in these scripts. It is not a smoothing or climatology operation.

### 8. Assumptions and limitations
- Fixed UTC+8 is assumed.
- Sample completeness is checked in some functions but the exact missing-sample tolerance varies by loader.

### 9. Methods-ready text
中文：所有日尺度诊断均按北京时间定义。UTC 或 valid_time 坐标先统一加 8 小时，再按 1 日窗口求平均或求和。Z500 和风场使用日平均，降水使用日累计。

English: Daily diagnostics were defined on Beijing Time calendar days. Time coordinates were shifted by +8 h before one-day resampling. Z500 and wind fields were averaged over each BJT day, whereas precipitation was accumulated over the BJT day.

### 10. Still needs confirmation/literature
- Confirm if manuscript should call this “BJT” or “local time (UTC+8)” consistently.

---

## 2.4 S2S accumulated precipitation step-difference and BJT calendar-day apportionment

### 1. Scientific purpose
To transform accumulated S2S precipitation into BJT daily precipitation totals without losing increments that cross calendar-day boundaries.

### 2. Code evidence
- Fig.7 `_apportion_accum_to_calendar_days_fig7` requires `step`, constructs BJT valid times, computes `inc = da.diff("step")`, optionally clips negative increments to zero, splits increments by overlap fraction with BJT calendar days, and groups by `bjt_date` (`cell_fig7_stageII_daily_synoptic_paper.py:594-634`).
- Fig.9 uses the same logic in `_apportion_accum_to_calendar_days` and `process_s2s_bjt_daily_accum`, then selects `ANALYSIS_DATES` (`cell_leadtime_total_hotcold_composite_maps.py:467-515`).

### 3. Current code's exact definition
- Applies only to accumulated S2S fields with a `step` coordinate.
- Increment between steps \(i-1\) and \(i\) is computed by differencing accumulated values.
- If `force_non_negative=True`, negative increments are set to 0.
- Each increment is split among BJT dates in proportion to elapsed seconds falling in each day.
- The apportioned pieces are summed by `bjt_date` and then converted to mm if output is TP.

### 4. Mathematical formula
Let \(C_i\) be accumulated precipitation at forecast step \(i\), with BJT valid interval \([s_i,e_i]\). Increment:

\[
\Delta C_i = C_i - C_{i-1}.
\]

If nonnegative clipping is enabled:

\[
\Delta C_i^{+} = \max(\Delta C_i, 0).
\]

For BJT day \(d\), overlap fraction:

\[
f_{i,d} = \frac{|[s_i,e_i]\cap d|}{e_i-s_i}.
\]

Daily total:

\[
P_d = \sum_i f_{i,d}\Delta C_i^{+}.
\]

### 5. Method principle
General principle: accumulated forecast fields must be differenced before aggregation. Current implementation assumes each step increment is uniformly distributed over its valid interval when partitioning across BJT days.

### 6. Output and interpretation
Output is daily S2S precipitation in mm day^-1. It is not an instantaneous precipitation rate.

### 7. Difference from adjacent methods
This differs from simply resampling accumulated values; it first converts accumulations to increments.

### 8. Assumptions and limitations
- Uniform distribution within forecast-step intervals is assumed for splitting.
- Negative increments are clipped when `force_non_negative=True`, which may hide small numerical resets.

### 9. Methods-ready text
中文：S2S 地表降水为累计量。代码先对相邻 forecast step 做差得到增量降水，再根据每个增量时间段与北京时间日历日的重叠比例分配到对应日期，最后按 BJT 日期求和。为避免累计量重置或数值噪声导致负降水，启用时负增量被置为 0。

English: S2S surface precipitation was treated as an accumulated field. Consecutive forecast steps were differenced to obtain incremental precipitation, and each increment was apportioned to BJT calendar days according to the fractional temporal overlap between the forecast-step interval and the day. Negative increments were set to zero when the non-negative option was enabled.

### 10. Still needs confirmation/literature
- Confirm ECMWF accumulation reset conventions for the specific S2S files.
- Cite ECMWF S2S/GRIB documentation for accumulated TP handling.

---

## 2.5 Spatial domain selection, cos(latitude) means, and Hovmöller latitude-band averaging

### 1. Scientific purpose
To define a common geographical analysis window and reduce 2-D fields to time-longitude Hovmöller diagnostics over the North China latitude band.

### 2. Code evidence
- Fig.7 `MAP_EXTENT`, `DATA_EXTENT`, and `NCHN_BOX` are defined as fixed geographic boxes (`cell_fig7_stageII_daily_synoptic_paper.py:35-37`).
- Fig.8 defines `LAT_BAND=(35,44)`, `LON_RANGE=(90,135)`, and `DATA_EXTENT=(88,137,33,46)` (`cell_hovmoller_tp_z500_obs_s2s_stageII.py:30-42`).
- `_latband_hov` selects the longitude/latitude domain and computes a cosine-latitude weighted mean over latitude (`cell_hovmoller_tp_z500_obs_s2s_stageII.py:188-199`).
- Cell Y `_area_mean` also uses cosine-latitude weights for box means (`cell_y_era5_z500_uv850_tp_daily_evolution.py:104-108`).

### 3. Current code's exact definition
- Hovmöller latitude band: 35–44N.
- Hovmöller longitude range: 90–135E.
- `weights = cos(latitude)` and weighted mean over latitude.
- For area means in Cell Y, mean over lat and lon weighted by cos(lat).

### 4. Mathematical formula
For Hovmöller field \(H(t,\lambda)\) from gridded field \(A(t,\phi,\lambda)\):

\[
H(t,\lambda) = \frac{\sum_{\phi \in [35,44]} A(t,\phi,\lambda)\cos\phi}{\sum_{\phi \in [35,44]}\cos\phi}.
\]

For box area mean over \(B\):

\[
\bar{A}_B(t) = \frac{\sum_{(\phi,\lambda)\in B} A(t,\phi,\lambda)\cos\phi}{\sum_{(\phi,\lambda)\in B}\cos\phi}.
\]

### 5. Method principle
Cosine weighting approximates grid-cell area for regular latitude-longitude grids. Current code applies this weighting where explicitly called; not every plotted map is area averaged.

### 6. Output and interpretation
Hovmöller outputs have dimensions time/date by longitude and units inherited from TP/Z500. The Hovmöller compresses meridional variations into a single latitude-band mean.

### 7. Difference from adjacent methods
Area/Hovmöller averaging differs from spatial compositing across ensemble members. The former reduces spatial dimensions; the latter averages members.

### 8. Assumptions and limitations
- Assumes regular lat-lon geometry and that cos(lat) weighting is adequate.
- Meridional displacement within 35–44N is not retained in Hovmöller plots.

### 9. Methods-ready text
中文：传播诊断采用 35–44°N 纬带平均的 Hovmöller 图。纬向平均使用 cos(latitude) 作为面积权重，并保留 90–135°E 的经度维度。

English: The propagation diagnostic was constructed as a time–longitude Hovmöller diagram averaged over 35–44°N. Latitude averaging used cos(latitude) weights, and longitudes from 90°E to 135°E were retained.

### 10. Still needs confirmation/literature
- Confirm whether the manuscript should call the box NCHN or HCNC consistently.

---

## 2.6 Total-stage Hot17/Cold17 ensemble-member grouping

### 1. Scientific purpose
To compare circulation and precipitation structures between S2S members classified as relatively hot or cold according to a precomputed Total-period Tmax grouping.

### 2. Code evidence
- Fig.7 uses `GROUP_FILE = /data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv`, `ABS_GROUPS=[Hot17, Cold17]`, `DIFF_GROUP=Hot17_minus_Cold17`, and labels containing `Total` (`cell_fig7_stageII_daily_synoptic_paper.py:48-65`).
- Fig.7 grouping parser filters by init date and Total stage and validates 17 unique members per group (`cell_fig7_stageII_daily_synoptic_paper.py:378-456`).
- Fig.9 uses `GROUPING_STAGE="Total"`, four init dates, and the same group file path (`cell_leadtime_total_hotcold_composite_maps.py:24-49`).
- Fig.9 parser validates 17 members per init/group and member IDs 0–50 (`cell_leadtime_total_hotcold_composite_maps.py:240-315`).

### 3. Current code's exact definition
- The current scripts **do not compute** the sorting variable used to define Hot17/Cold17. They read a grouping CSV.
- Stage is filtered by normalized text matching `Total` or `Total-stage` variants.
- Groups used in Fig.7/Fig.9 formal outputs are `Hot17` and `Cold17`. Fig.8 code contains `Middle17` in group order for diagnostic figures, but the main from-cache Fig.8 uses observation, Total Hot17, Total Hot17−Cold17, and Total Cold17.
- For each init date and group, exactly 17 members are required; duplicate or out-of-range member IDs raise errors.

### 4. Mathematical formula
Let \(G_{g,i}\) be the set of member IDs for group \(g\in\{Hot17,Cold17\}\) and initialization \(i\), read from the CSV:

\[
|G_{g,i}| = 17, \quad G_{g,i}\subset\{0,1,\ldots,50\}.
\]

### 5. Method principle
General principle: conditional composites compare groups selected by a predefined criterion. Current implementation reads group membership rather than deriving it from fields in these scripts.

### 6. Output and interpretation
Output is a member list and group audit table. A Hot17/Cold17 label in these scripts means “member set read from Total-stage grouping file,” not necessarily a ranking recomputed within the plotting script.

### 7. Difference from adjacent methods
Grouping is separate from compositing. Grouping selects members; compositing averages selected members.

### 8. Assumptions and limitations
- The grouping file must be correct and match the intended Tmax metric.
- Because groups are based on target behavior, composite contrasts are conditional diagnostics and should not be interpreted as independent causal effects.
- Multi-init Fig.9 determines member sets separately for each initialization because the parser loops over each `INIT_DATES` value.

### 9. Methods-ready text
中文：Hot17 和 Cold17 成员不是在绘图脚本中重新排序得到，而是从预先生成的 Total-stage Tmax 分组表读取。每个起报日期分别筛选 stage = Total 的 Hot17 和 Cold17 成员，并强制检查每组 17 个、成员编号位于 0–50。

English: Hot17 and Cold17 members were read from a precomputed Total-stage Tmax-based grouping table rather than recalculated in the plotting scripts. For each initialization date, rows with stage equal to Total were selected, and exactly 17 unique members were required for each of the Hot17 and Cold17 groups.

### 10. Still needs confirmation/literature
- Code cannot confirm the exact Tmax sorting variable or whether it is Tmax error, Tmax anomaly, or another regional Tmax statistic. This must be audited in the grouping-generation script.
- Suggested search terms: “tmax_member_hotcold_group_by_stage generation script”, “Total-stage Tmax grouping definition”.

---

## 2.7 CF/PF merge and 51-member S2S ensemble handling

### 1. Scientific purpose
To combine ECMWF S2S control and perturbed forecasts into a consistent 51-member ensemble before group selection.

### 2. Code evidence
- Fig.7 `_concat_cf_pf_fig7` normalizes member dimensions, concatenates CF/PF along `member`, assigns sequential member IDs, and raises if neither stream is loaded (`cell_fig7_stageII_daily_synoptic_paper.py:638-643`).
- Fig.7 loading checks `member_count == 51` after CF/PF merge (`cell_fig7_stageII_daily_synoptic_paper.py:670-678` from search output context).
- Fig.9 analogous `_concat_cf_pf` and `load_s2s_ensemble_with_cf` normalize member dimensions and print/check member count (`cell_leadtime_total_hotcold_composite_maps.py:354-376`, `428-437` from search context).

### 3. Current code's exact definition
- If no member-like dimension exists, the control stream is expanded as member 0 and perturbed stream as member 1 before offsetting.
- Member coordinate is assigned with offset 0 for CF and 1 for PF, yielding expected member IDs 0–50 after concatenation.
- Later group selection assumes member IDs in the grouping file refer to this 0–50 coordinate.

### 4. Mathematical formula
For control field \(F^{CF}\) and perturbed field \(F^{PF}\), the ensemble field is

\[
F_m = \operatorname{concat}(F^{CF},F^{PF})_m, \quad m=0,1,\ldots,50.
\]

### 5. Method principle
General principle: control and perturbed streams must share a common member coordinate for ensemble operations. Current code constructs this coordinate explicitly.

### 6. Output and interpretation
Output is an S2S field with dimension `member`. It represents the merged 51-member ensemble.

### 7. Difference from adjacent methods
CF/PF merge defines ensemble membership; Hot17/Cold17 grouping selects subsets of this member dimension.

### 8. Assumptions and limitations
- Assumes CF/PF file contents correspond to the same initialization and variable.
- Assumes member offset convention matches grouping CSV.

### 9. Methods-ready text
中文：S2S 控制成员和扰动成员在进入分组前合并为统一的 member 维度。代码将 CF/PF 的成员维度标准化，并赋予 0–50 的成员编号，后续 Hot17/Cold17 分组直接按该编号选择成员。

English: Control and perturbed S2S streams were merged onto a common member coordinate before compositing. Member indices were standardized to 0–50, and the Hot17/Cold17 group lists were applied to this merged ensemble dimension.

### 10. Still needs confirmation/literature
- ECMWF S2S documentation should support the 51-member configuration and CF/PF convention.

---

## 2.8 Group composites and Hot17 − Cold17 differences

### 1. Scientific purpose
To compare mean precipitation and circulation structures between the hot and cold forecast-member groups and to show their contrast.

### 2. Code evidence
- Fig.9 `group_mean` selects members and averages over `member`; `build_total_composites_for_init` computes period-mean TP and Z500 for Hot17/Cold17 and then subtracts Cold17 from Hot17 (`cell_leadtime_total_hotcold_composite_maps.py:529-548`).
- Fig.9 QC records `sign_convention = Hot17_minus_Cold17 = Hot17 - Cold17` (`cell_leadtime_total_hotcold_composite_maps.py:803-839`).
- Fig.9 cache export stores `tp_diff` and `z500_abs_diff` as Hot17-minus-Cold17 fields and metadata records sign convention and `z500_diff_type` (`cell_leadtime_total_hotcold_composite_maps.py:1001-1043`).
- Fig.7 and Fig.8 cache exports store analogous `tp_diff`/`z500_diff` and `tp_diff_hov`/`z500_diff_hov` fields.

### 3. Current code's exact definition
- Group mean is calculated after daily aggregation and, for Fig.9, after period averaging.
- Fig.9: period mean over 2023-06-14 to 2023-06-24 is computed for each member, then group mean over selected members.
- Difference direction is explicitly Hot17 − Cold17.
- No p-value or significance mask is applied in this window.

### 4. Mathematical formula
For variable \(A_{m,t}(\phi,\lambda)\), group mean for group \(g\):

\[
\bar{A}_{g}(t,\phi,\lambda) = \frac{1}{|G_g|}\sum_{m\in G_g} A_{m,t}(\phi,\lambda).
\]

For period mean in Fig.9:

\[
\bar{A}_{g}^{P}(\phi,\lambda) = \frac{1}{N_t}\sum_{t\in P}\bar{A}_{g}(t,\phi,\lambda).
\]

Difference:

\[
\Delta A = \bar{A}_{Hot17} - \bar{A}_{Cold17}.
\]

### 5. Method principle
Composite analysis averages fields conditional on group membership. Current implementation uses deterministic group lists and arithmetic means; no resampling uncertainty estimate is computed.

### 6. Output and interpretation
- TP composite: mm day^-1.
- Z500 absolute composite: gpm.
- Difference: same units as variable.
Differences quantify association with group membership; they are not causal impacts.

### 7. Difference from adjacent methods
Composite difference differs from anomaly: Hot17−Cold17 compares groups, whereas anomaly subtracts a climatology.

### 8. Assumptions and limitations
- Group selection is based on target Tmax behavior, so composites are conditional on the outcome-related grouping.
- Sample size is 17 per group, so results may be sensitive to member selection.
- No spatial significance method is implemented in the current window.

### 9. Methods-ready text
中文：对每个起报日期，先按成员维度分别对 Hot17 和 Cold17 求集合平均，再计算 Hot17 − Cold17 差值。差值方向在所有图和 QC 中均定义为 Hot17_minus_Cold17 = Hot17 − Cold17。

English: For each initialization, Hot17 and Cold17 composites were computed as arithmetic means over the corresponding 17 ensemble members. Difference fields were defined consistently as Hot17 minus Cold17 for all variables and diagnostics.

### 10. Still needs confirmation/literature
- If manuscript discusses significance, a separate significance-testing implementation must be audited; none is found here.

---

## 2.9 ERA5 and S2S Z500 anomaly diagnostics

### 1. Scientific purpose
To diagnose circulation departures from a climatological or reforecast reference state in daily evolution diagnostics. This is distinct from Fig.9, which currently uses absolute Z500.

### 2. Code evidence
- Cell Y defines `CLIM_YEARS = 2003..2022` and `REQUIRE_Z500_CLIM_FOR_ANOMALY = True` (`cell_y_era5_z500_uv850_tp_daily_evolution.py:22-25`).
- Cell Y loads each year’s June Z500 file, selects same month-day target dates, assigns those historical dates to `TARGET_DATES`, concatenates over `clim_year`, and averages (`cell_y_era5_z500_uv850_tp_daily_evolution.py:250-301`).
- Cell Y computes `z500_anom = z500_2023 - z500_clim` (`cell_y_era5_z500_uv850_tp_daily_evolution.py:441-449`).
- Cell Z builds S2S realtime and hindcast daily Z500; hindcast samples are averaged into `z500_clim`, and `z500_anom = member_mean(realtime Z500) - z500_clim` (`cell_z_s2s_z500_uv850_tp_qflux_daily_evolution.py:444-463`).

### 3. Current code's exact definition
- ERA5 anomaly: 2023 daily Z500 minus 2003–2022 same-month-day daily climatology for target dates 2023-06-17 to 2023-06-20 in Cell Y.
- S2S anomaly in Cell Z: realtime member-mean Z500 minus sample-mean hindcast Z500, selected on `TARGET_DATES`.
- Fig.9 lead-time cache explicitly uses `z500_abs`, not anomaly, and stores `z500_type = absolute` (`cell_leadtime_total_hotcold_composite_maps.py:1001-1043`).

### 4. Mathematical formula
ERA5 climatology:

\[
Z^{clim}(d,\phi,\lambda)=\frac{1}{20}\sum_{y=2003}^{2022}Z_y(mmdd(d),\phi,\lambda).
\]

ERA5 anomaly:

\[
Z'(d,\phi,\lambda)=Z_{2023}(d,\phi,\lambda)-Z^{clim}(d,\phi,\lambda).
\]

S2S anomaly in Cell Z:

\[
Z'_{S2S}(d,\phi,\lambda)=\frac{1}{M}\sum_m Z^{rt}_{m,d}(\phi,\lambda)-\frac{1}{S}\sum_s Z^{hc}_{s,d}(\phi,\lambda).
\]

### 5. Method principle
Anomaly diagnostics remove a baseline seasonal/day-specific reference. Current code contains both ERA5 historical climatology and S2S reforecast-climatology variants.

### 6. Output and interpretation
Output is Z500 anomaly in gpm. It represents departure from a historical or reforecast baseline, not absolute circulation height.

### 7. Difference from adjacent methods
Absolute Z500 contours in Fig.7/Fig.8/Fig.9 show full height fields; anomaly fields show departures. Fig.9 currently uses absolute Z500, not anomaly.

### 8. Assumptions and limitations
- ERA5 climatology depends on 2003–2022 June files and exact same month-day selection.
- S2S reforecast climatology depends on hindcast samples and their alignment with target dates.
- No significance test for anomalies is found in current window.

### 9. Methods-ready text
中文：在日演变诊断中，ERA5 Z500 anomaly 定义为 2023 年对应日期日平均 Z500 减去 2003–2022 年同月同日 ERA5 日平均 Z500 的平均值。S2S anomaly 诊断中，realtime 成员平均 Z500 减去 hindcast/reforecast 样本平均 Z500。需要注意的是，当前 Fig.9 lead-time composite 使用的是 absolute Z500，而不是 anomaly。

English: In the daily-evolution diagnostics, ERA5 Z500 anomalies were computed as the 2023 daily Z500 field minus the mean ERA5 daily field for the same calendar dates over 2003–2022. In the S2S anomaly diagnostic, the realtime ensemble-mean Z500 field was compared with the corresponding hindcast sample mean. The current Fig.9 lead-time composite, however, uses absolute Z500 rather than Z500 anomaly.

### 10. Still needs confirmation/literature
- Confirm whether anomaly diagnostics are included in the final Methods or only supplement.
- Cite ERA5 and ECMWF reforecast documentation.

---

## 2.10 Fig.7 daily synoptic maps

### 1. Scientific purpose
To show day-by-day synoptic evolution during 18–20 June for observed precipitation/circulation and the S2S Total Hot17/Cold17 member groups.

### 2. Code evidence
- Fig.7 `TARGET_DATES` are 2023-06-18, 2023-06-19, 2023-06-20 (`cell_fig7_stageII_daily_synoptic_paper.py:30`).
- Observed row uses MSWEP TP and ERA5 Z500/UV850 according to `load_observed_daily_fields_fig7` docstring and loaders (`cell_fig7_stageII_daily_synoptic_paper.py:327-330`).
- Main and appendix stems indicate total-group outputs for init 2023-06-12 and 2023-06-08 (`cell_fig7_stageII_daily_synoptic_paper.py:67-73`).
- Cache export writes `tp_obs`, `z500_obs`, `tp_hot`, `z500_hot`, `tp_cold`, `z500_cold`, `tp_diff`, and `z500_diff` for Fig.7 (`cell_fig7_stageII_daily_synoptic_paper.py:1191-1216` from previous code scan).

### 3. Current code's exact definition
- Fig.7 main: dates 18–20 June, main init 2023-06-12, observation row plus Total Hot17, Total Cold17, and Hot17−Cold17.
- Appendix: init 2023-06-08.
- Optional vector version uses UV850; no-vector version plots TP shading and Z500 contours only.

### 4. Mathematical formula
Daily group composite:

\[
\bar{A}_{g,t}(\phi,\lambda)=\frac{1}{17}\sum_{m\in G_g} A_{m,t}(\phi,\lambda),\quad t\in\{18,19,20\,Jun\}.
\]

Difference:

\[
\Delta A_t = \bar{A}_{Hot17,t}-\bar{A}_{Cold17,t}.
\]

### 5. Method principle
Daily synoptic maps are descriptive composites. They diagnose spatial circulation/precipitation structures associated with member groups.

### 6. Output and interpretation
Outputs are maps of TP (mm day^-1) and Z500 (gpm). Differences show group contrast, not causal effects.

### 7. Difference from adjacent methods
Fig.7 daily maps retain latitude-longitude structure for individual dates, unlike Fig.8 Hovmöllers (time-longitude) or Fig.9 period means.

### 8. Assumptions and limitations
- Only three key dates are shown.
- No p-value mask is implemented in the current window.

### 9. Methods-ready text
中文：Fig.7 诊断 2023 年 6 月 18–20 日的逐日天气尺度演变。观测行使用 MSWEP 日累计降水和 ERA5 500 hPa 位势高度；S2S 行使用 2023-06-12 起报的 Total Hot17、Total Cold17 以及 Hot17−Cold17 差值。

English: Fig.7 summarizes the daily synoptic evolution during 18–20 June 2023. The observational row combines MSWEP daily precipitation with ERA5 500-hPa geopotential height, whereas the S2S rows show Total Hot17, Total Cold17, and Hot17-minus-Cold17 composites for the 12 June initialization.

### 10. Still needs confirmation/literature
- Confirm whether the vector and no-vector versions are both described or only the final manuscript version.

---

## 2.11 Fig.8 Hovmöller propagation diagnostic

### 1. Scientific purpose
To diagnose time-longitude propagation of precipitation and Z500 structures during 14–24 June.

### 2. Code evidence
- Fig.8 `WINDOWS["0614_0624"]` is 2023-06-14 to 2023-06-24 (`cell_hovmoller_tp_z500_obs_s2s_stageII.py:20-34`).
- Lat band, longitude range, and reference lines are configured as `LAT_BAND=(35,44)` and `LON_RANGE=(90,135)` (`cell_hovmoller_tp_z500_obs_s2s_stageII.py:30-42`).
- `_latband_hov` computes cosine-weighted mean across latitude and returns `bjt_date, longitude` (`cell_hovmoller_tp_z500_obs_s2s_stageII.py:188-199`).
- Fig.8 cache export stores TP/Z500 Hovmöller fields and metadata including `hovmoller_lat_band` and `vertical_reference_lines` (`cell_hovmoller_tp_z500_obs_s2s_stageII.py:1280-1325` from previous code scan).

### 3. Current code's exact definition
- Observation: MSWEP TP Hovmöller + ERA5 Z500 Hovmöller.
- S2S: Total Hot17, Total Cold17, and Hot17−Cold17 for init 2023-06-12 in the main from-cache plot. Appendix/cache also considers 2023-06-08 for levels.
- Time axis uses BJT dates.
- Vertical dashed lines at 114E and 119E mark the NCHN longitude range.

### 4. Mathematical formula
For precipitation Hovmöller:

\[
P^{hov}_{g}(t,\lambda)=\frac{\sum_{\phi\in[35,44]}\bar{P}_{g}(t,\phi,\lambda)\cos\phi}{\sum_{\phi\in[35,44]}\cos\phi}.
\]

For difference:

\[
\Delta P^{hov}(t,\lambda)=P^{hov}_{Hot17}(t,\lambda)-P^{hov}_{Cold17}(t,\lambda).
\]

### 5. Method principle
Hovmöller diagrams compress a spatial field to one horizontal coordinate plus time, useful for propagation diagnostics. Current implementation averages over latitude and retains longitude.

### 6. Output and interpretation
Outputs are time-longitude fields of precipitation (mm day^-1) and Z500 (gpm). They indicate co-evolution/progression, not causal propagation mechanisms by themselves.

### 7. Difference from adjacent methods
Fig.8 differs from Fig.7 maps by sacrificing meridional spatial detail to emphasize time-longitude evolution.

### 8. Assumptions and limitations
- Latitude-band average may hide north-south displacement.
- Propagation diagnosis is descriptive unless supplemented by dynamical analysis.

### 9. Methods-ready text
中文：传播诊断采用 2023 年 6 月 14–24 日的 Hovmöller 图。降水和 Z500 先在 35–44°N 纬带内按 cos(latitude) 加权平均，保留 90–135°E 经度维度，并在 114°E 和 119°E 标记华北关键经度范围。

English: The propagation diagnostic used time–longitude Hovmöller diagrams for 14–24 June 2023. Precipitation and Z500 were averaged over 35–44°N using cos(latitude) weights, retaining the 90–135°E longitude dimension. Vertical reference lines at 114°E and 119°E mark the North China longitude band.

### 10. Still needs confirmation/literature
- Confirm whether “propagation” should be described as diagnostic/visual rather than inferred phase speed.

---

## 2.12 Fig.9 multi-init Total-period lead-time composites

### 1. Scientific purpose
To assess whether Total-stage Hot17/Cold17 precipitation and 500-hPa circulation composites depend on S2S lead time/initialization date.

### 2. Code evidence
- Fig.9 uses `INIT_DATES = 2023-06-01, 06-05, 06-08, 06-12`, `ANALYSIS_DATES = 2023-06-14 to 2023-06-24`, and `GROUPING_STAGE = Total` (`cell_leadtime_total_hotcold_composite_maps.py:24-33`).
- `build_daily_s2s_member_fields` loads S2S TP and realtime Z500, processes daily TP with step apportionment and Z500 with daily mean, and selects analysis dates (`cell_leadtime_total_hotcold_composite_maps.py:495-519`).
- `_period_mean` averages over `bjt_date`; `group_mean` averages over selected members; `build_total_composites_for_init` computes Hot17/Cold17 and Hot17−Cold17 period-mean TP/Z500_abs (`cell_leadtime_total_hotcold_composite_maps.py:529-548`).
- Reference fields are MSWEP period-mean TP and ERA5 period-mean absolute Z500 (`cell_leadtime_total_hotcold_composite_maps.py:562-590`).
- Cache export stores `tp_hot`, `z500_abs_hot`, `tp_cold`, `z500_abs_cold`, `tp_diff`, `z500_abs_diff`, `tp_obs_ref`, and `z500_abs_obs_ref` (`cell_leadtime_total_hotcold_composite_maps.py:1001-1009`).

### 3. Current code's exact definition
- For each init, daily TP/Z500 are selected for 14–24 June BJT.
- A total-period mean is computed over BJT dates.
- Group means are computed over Total Hot17/Cold17 members for that init.
- Difference is Hot17−Cold17.
- Z500 is absolute geopotential height, not anomaly.

### 4. Mathematical formula
For period \(P=\{14,\ldots,24\,Jun\}\):

\[
A_{m}^{P}(\phi,\lambda)=\frac{1}{|P|}\sum_{t\in P}A_{m,t}(\phi,\lambda).
\]

Group composite for init \(i\):

\[
\bar{A}_{g,i}^{P}(\phi,\lambda)=\frac{1}{17}\sum_{m\in G_{g,i}} A_{m,i}^{P}(\phi,\lambda).
\]

Difference:

\[
\Delta A_i^{P}=\bar{A}_{Hot17,i}^{P}-\bar{A}_{Cold17,i}^{P}.
\]

### 5. Method principle
The method compares conditional group composites across forecast initialization dates. It is a descriptive lead-time sensitivity diagnostic.

### 6. Output and interpretation
- TP: period-mean mm day^-1.
- Z500_abs: period-mean gpm.
- Difference fields: Hot17−Cold17 in same units.
It does not estimate model skill or causal lead-time effects.

### 7. Difference from adjacent methods
Unlike Fig.7 daily maps, Fig.9 averages over 14–24 June. Unlike anomaly diagnostics, Fig.9 uses absolute Z500 contours.

### 8. Assumptions and limitations
- Group membership is independently read for each init; this means Hot17 for one init is not guaranteed to contain the same members as Hot17 for another init.
- No significance test is implemented.

### 9. Methods-ready text
中文：Fig.9 比较四个起报日期的 Total-stage Hot17/Cold17 合成。对每个起报日期，先将 S2S 日降水和 Z500 选取 2023 年 6 月 14–24 日并求时段平均，再按该起报日期的 Total Hot17/Cold17 成员分别求集合平均。差值定义为 Hot17 − Cold17。Z500 使用 absolute geopotential height，不使用 anomaly。

English: Fig.9 compares Total-stage Hot17 and Cold17 composites across four initialization dates. For each initialization, daily S2S precipitation and absolute Z500 were averaged over 14–24 June 2023 and then averaged across the 17 members in each group. Difference fields were defined as Hot17 minus Cold17. Z500 in this diagnostic is absolute geopotential height rather than an anomaly.

### 10. Still needs confirmation/literature
- Confirm whether the term “lead-time dependence” is acceptable or should be “initialization dependence”.

---

## 2.13 Moisture flux and MFC diagnostics

### 1. Scientific purpose
To diagnose moisture-transport-related fields accompanying precipitation evolution.

### 2. Code evidence
- Cell X computes q-flux from ERA5 q/u/v fields, counts BJT samples, and computes MFC as `-compute_divergence_lonlat(qu_d, qv_d) * 1e5` (`cell_x_era5_mswep_daily_evolution.py:187-217`, `340-341` from search output).
- Cell Z computes `qu850 = q850 * u850` and `qv850 = q850 * v850`, sets units, and stores daily q-flux fields (`cell_z_s2s_z500_uv850_tp_qflux_daily_evolution.py:405-425`).

### 3. Current code's exact definition
- q-flux: q850 multiplied by u850/v850.
- MFC in Cell X: negative horizontal divergence of q-flux, scaled by 1e5.
- These diagnostics are not part of the final Fig.7–Fig.9 from-cache scripts but are in current window.

### 4. Mathematical formula
\[
Q_u = q u,\quad Q_v = q v.
\]

\[
MFC = -\nabla_h\cdot(q\mathbf{V})\times10^5.
\]

### 5. Method principle
Moisture flux describes wind-weighted specific humidity transport; moisture flux convergence approximates horizontal moisture convergence.

### 6. Output and interpretation
q-flux units are set to kg kg^-1 m s^-1. MFC is scaled by 1e5 in plotting/QC. Scaling is for visualization/diagnostic convenience.

### 7. Difference from adjacent methods
q-flux/MFC is a process diagnostic, not a member grouping or forecast-error metric.

### 8. Assumptions and limitations
- Divergence on latitude-longitude grid depends on finite-difference implementation.
- Vertical integration is not indicated; it is 850-hPa diagnostic only.

### 9. Methods-ready text
中文：水汽通量诊断使用 850 hPa 比湿与风场相乘得到 qU 和 qV。MFC 定义为 −div(qV)，并在图中乘以 1e5 作为缩放量。

English: Moisture-flux diagnostics were computed at 850 hPa as qU and qV. Moisture-flux convergence was diagnosed as the negative horizontal divergence of the q-flux vector and was scaled by 1e5 for plotting.

### 10. Still needs confirmation/literature
- Confirm whether MFC/q-flux appears in final Methods or only supplementary diagnostics.

---

## 2.14 Cache export and QC

### 1. Scientific purpose
To separate expensive science processing from style-only figure plotting and to provide reproducible plot-ready fields.

### 2. Code evidence
- Fig.7 cache export writes fields such as `tp_obs`, `z500_obs`, `tp_hot`, `z500_hot`, `tp_cold`, `z500_cold`, `tp_diff`, and `z500_diff`, plus TP levels and metadata (`cell_fig7_stageII_daily_synoptic_paper.py:1191-1235` from previous code scan).
- Fig.8 cache export writes Hovmöller fields and metadata including `z500_type`, `z500_diff_type`, `hovmoller_lat_band`, and reference lines (`cell_hovmoller_tp_z500_obs_s2s_stageII.py:1280-1325` from previous code scan).
- Fig.9 cache export writes lead-time fields and metadata including grouping file, grouping stage, sign convention, and z500 type (`cell_leadtime_total_hotcold_composite_maps.py:968-1043`).
- Fig.9 QC includes TP max location and NCHN mean/max fields (`cell_leadtime_total_hotcold_composite_maps.py:803-839`).

### 3. Current code's exact definition
- Cache files are NetCDF files under `/data1/huangy/fig6/NC/v12/paper_figures/cache/`.
- QC CSV includes field-level statistics such as min/max/mean/nan_count/finite_fraction; Fig.9 additionally records TP maximum location and NCHN summary.
- From-cache scripts read only cached NetCDF and do not recompute group membership.

### 4. Mathematical formula
QC finite fraction:

\[
f_{finite}=\frac{\#\{\mathrm{finite}\;values\}}{\#\{\mathrm{all}\;values\}}.
\]

Domain mean:

\[
\bar{A}_{domain}=\frac{1}{N}\sum_i A_i
\]

where implemented as simple nan-aware mean unless separately weighted.

### 5. Method principle
Cache export preserves final plot-ready fields, enabling style changes without rerunning upstream data processing.

### 6. Output and interpretation
Cache values are the science fields for plotting. QC values are reproducibility/sanity diagnostics, not primary scientific results unless explicitly reported.

### 7. Difference from adjacent methods
Cache export stores computed fields; from-cache plotting adjusts visual appearance only.

### 8. Assumptions and limitations
- Cache validity depends on upstream processing correctness.
- QC cannot prove scientific correctness; it detects obvious dimensional/NaN problems.

### 9. Methods-ready text
中文：为将科学处理与绘图排版解耦，最终绘图所需的日尺度、Hovmöller 和时段平均场被导出为 NetCDF cache。后续绘图脚本仅读取 cache，不重新读取原始 ERA5/MSWEP/S2S 数据，也不重新计算分组或合成。

English: To separate science processing from figure layout revisions, plot-ready daily, Hovmöller, and period-mean fields were exported to NetCDF caches. The final plotting scripts read only these caches and do not reload raw ERA5/MSWEP/S2S data or recompute group membership and composites.

### 10. Still needs confirmation/literature
- Methods may mention the cache workflow only if relevant to reproducibility; it is not a scientific method per se.

---

# 3. Methods section suggested structure

## 2.1 Data and preprocessing
- ERA5, MSWEP, ECMWF S2S inputs.
- Variable identification, coordinate normalization, pressure-level selection.
- Unit conversion for TP and Z500.
- BJT daily aggregation.
- Spatial domains and NCHN box.

## 2.2 Event windows and ensemble-member grouping
- Fig.7 dates: 18–20 June.
- Fig.8 propagation window: 14–24 June.
- Fig.9 lead-time/initialization window: 14–24 June and four init dates.
- Total-stage Hot17/Cold17 grouping, 17 members per group, per-init grouping.

## 2.3 S2S precipitation and circulation preprocessing
- CF/PF merge to 51 members.
- Accumulated TP step differencing and BJT apportionment.
- Daily mean Z500 and wind fields.

## 2.4 Composite and difference diagnostics
- Group means.
- Hot17 − Cold17 sign convention.
- Daily synoptic maps.
- Hovmöller latitude-band mean.
- Lead-time period-mean composites.

## 2.5 Climatology and anomaly diagnostics
- ERA5 2003–2022 same-date Z500 climatology in Cell Y.
- S2S realtime-minus-hindcast Z500 anomaly in Cell Z.
- Explicitly state Fig.9 uses absolute Z500, not anomaly.

## 2.6 Process diagnostics
- UV850 vectors where used.
- q-flux and MFC where included in supplementary/daily-evolution figures.

## 2.7 Robustness and quality control
- Member-count validation.
- Date-coverage checks.
- Cache QC fields.
- Missing-file behavior.

Methods for MLR, LMG, Johnson relative weights, LOFO, Bias, RMSE, partial fitted component, KDE and historical rank should be placed in separate sections only after auditing the scripts that actually implement them; they are not implemented in the current spatial-propagation window.

---

# 4. Formula and symbol table

| Symbol | Meaning |
|---|---|
| \(t\) | BJT daily date |
| \(d\) | BJT calendar day |
| \(m\) | S2S ensemble member index |
| \(G_g\) | Set of member IDs in group \(g\) |
| \(g\) | Group, e.g., Hot17 or Cold17 |
| \(i\) | S2S initialization date |
| \(P\) | Analysis period, e.g., 2023-06-14 to 2023-06-24 |
| \(\phi\) | Latitude |
| \(\lambda\) | Longitude |
| \(A\) | Generic variable field |
| \(P_d\) | Daily precipitation total for day \(d\) |
| \(Z_{500}\) | 500-hPa geopotential height in gpm |
| \(\Phi_{500}\) | 500-hPa geopotential in m^2 s^-2 |
| \(g_0\) | Standard gravity 9.80665 m s^-2 |
| \(\Delta A\) | Hot17 − Cold17 difference |
| \(Z'\) | Z500 anomaly relative to a climatology or reforecast baseline |
| \(Q_u,Q_v\) | 850-hPa moisture flux components qU and qV |

---

# 5. Interpretation boundaries

- Hot17−Cold17 differences are conditional composite contrasts, not causal effects.
- No p<0.05 spatial significance test is implemented in the current window; do not describe stippling or significance unless another script is audited.
- Fig.9 uses absolute Z500; do not call Fig.9 contours anomalies.
- Total-stage grouping is read from a CSV; the sorting variable and ranking algorithm are not implemented in these scripts.
- BJT daily fields are local-day aggregates after a fixed +8 h shift.
- S2S TP daily values are produced from accumulated step differences, not from direct daily rate variables.
- Hovmöller diagnostics show time-longitude evolution after latitude-band averaging and should not be described as proof of dynamical propagation.
- q-flux/MFC diagnostics are process diagnostics but not causal attribution.
- Cache QC validates dimensions/NaNs and member counts but does not validate the physical correctness of input files.

---

# 6. Methods draft text

## 6.1 中文 Methods 草稿

本研究使用 ERA5、MSWEP 和 ECMWF S2S 集合预报资料诊断 2023 年 6 月华北高温事件期间降水和 500 hPa 环流结构。所有时间均转换为北京时间（BJT, UTC+8）后再进行日尺度聚合。降水使用 BJT 日累计，Z500 和风场使用 BJT 日平均。ERA5 或 S2S 中以 geopotential 形式存储的 500 hPa 位势通过除以 9.80665 m s^-2 转换为 geopotential meters。MSWEP 与 ERA5/S2S 保留各自原生网格；绘图 cache 中也使用 source-specific 的经纬度维度，未进行新的插值或 regrid。

S2S 降水为累计量时，先对相邻 forecast step 做差得到增量降水，再按照每个增量时间段与 BJT 日历日的重叠比例分配到对应日期，最后按 BJT 日期求和。控制成员和扰动成员在进入分组前合并为统一的 51 成员集合，成员编号为 0–50。

Hot17 和 Cold17 成员从预先生成的 Total-stage Tmax 分组表读取。每个起报日期分别筛选 stage = Total 的 Hot17 和 Cold17 成员，并要求每组恰好 17 个成员。脚本不重新计算 Tmax 排序变量。对每个 group，合成场定义为所选成员的算术平均；差值场统一定义为 Hot17 − Cold17。

Fig.7 展示 2023 年 6 月 18–20 日的日尺度天气形势。观测行使用 MSWEP 日降水和 ERA5 500 hPa 位势高度；S2S 行使用 2023-06-12 起报的 Total Hot17、Total Cold17 以及 Hot17−Cold17 合成差值。Fig.8 使用 2023 年 6 月 14–24 日的 Hovmöller 图诊断时间–经度演变；降水和 Z500 先在 35–44°N 内用 cos(latitude) 加权平均，并保留 90–135°E 经度维度。Fig.9 比较 2023-06-01、06-05、06-08 和 06-12 四个起报日期的 Total-stage 合成。对每个起报日期，先对 2023 年 6 月 14–24 日求时段平均，再按 Hot17/Cold17 成员求组平均。Fig.9 中 Z500 使用 absolute geopotential height，而不是 anomaly。

在单独的日演变诊断中，ERA5 Z500 anomaly 定义为 2023 年同日 Z500 减去 2003–2022 年同月同日 ERA5 日平均 Z500 的平均值；S2S anomaly 诊断中，realtime ensemble mean Z500 减去 hindcast/reforecast 样本平均 Z500。当前窗口未发现 p<0.05 空间显著性检验、MLR、LMG、Johnson relative weights、LOFO、Bias、RMSE、partial fitted component、KDE 或历史百分位排名的实现。

## 6.2 English JGR-style Methods draft

We used ERA5, MSWEP, and ECMWF S2S ensemble fields to diagnose precipitation and 500-hPa circulation structures during the June 2023 North China heatwave period. All daily diagnostics were defined on Beijing Time (BJT; UTC+8) calendar days. Precipitation was accumulated over each BJT day, whereas Z500 and wind fields were averaged over each BJT day. Geopotential fields stored in m^2 s^-2 were converted to geopotential meters by dividing by 9.80665 m s^-2. MSWEP, ERA5, and S2S fields were retained on their native grids; the plot-ready caches use source-specific latitude and longitude coordinates and do not introduce additional regridding.

For S2S accumulated precipitation, consecutive forecast steps were differenced to obtain incremental precipitation. Each increment was apportioned to BJT calendar days according to the fractional temporal overlap between the forecast-step interval and the day, and the apportioned increments were then summed by BJT date. Control and perturbed streams were merged onto a common 51-member coordinate before group selection.

Hot17 and Cold17 members were read from a precomputed Total-stage Tmax grouping table. For each initialization date, rows with stage equal to Total were selected, and exactly 17 members were required for each group. The grouping variable itself was not recalculated in these plotting scripts. Group composites were computed as arithmetic means over the selected members. Difference fields were consistently defined as Hot17 minus Cold17.

Fig.7 shows the daily synoptic evolution during 18–20 June 2023. The observational row combines MSWEP daily precipitation with ERA5 500-hPa geopotential height, while the S2S panels show Total Hot17, Total Cold17, and Hot17-minus-Cold17 composites for the 12 June initialization. Fig.8 diagnoses time–longitude evolution over 14–24 June using Hovmöller diagrams. Precipitation and Z500 were averaged over 35–44°N using cos(latitude) weights while retaining the 90–135°E longitude dimension. Fig.9 compares Total-stage composites for the 1, 5, 8, and 12 June initializations. For each initialization, daily fields were averaged over 14–24 June before computing Hot17 and Cold17 group means. Z500 in Fig.9 is absolute geopotential height rather than an anomaly.

In separate daily-evolution diagnostics, ERA5 Z500 anomalies were computed as 2023 daily Z500 minus the 2003–2022 ERA5 mean for the same calendar dates. In the S2S anomaly diagnostic, the realtime ensemble-mean Z500 field was compared with the hindcast/reforecast sample mean. No implementation of p<0.05 spatial significance testing, MLR, LMG, Johnson relative weights, LOFO, Bias, RMSE, partial fitted components, KDE, or historical percentile/rank diagnostics was found in the current spatial-propagation code window.

---

# 7. Needs-citation and needs-confirmation list

## 7.1 Code cannot confirm
- Exact Tmax sorting variable used to construct Hot17/Cold17 Total-stage grouping.
- Whether grouping was based on Tmax forecast error, anomaly, or raw Tmax.
- Whether any independent significance testing exists in another code window.
- Dataset product versions and retrieval dates.
- Whether MLR/LMG/Johnson/LOFO/partial-fitted components are implemented in another module.

## 7.2 Needs author confirmation
- Preferred terminology: North China box vs HCNC/NCHN.
- Whether Fig.9 should be described as “lead-time dependence” or “initialization dependence.”
- Whether q-flux/MFC diagnostics are Methods-relevant or supplementary only.
- Whether cache workflow should be described in Methods or only in reproducibility notes.

## 7.3 Needs original-method literature support
- ERA5 data citation.
- MSWEP data citation.
- ECMWF S2S/reforecast documentation.
- Hovmöller diagram methodology, if desired.
- Relative importance methods only if MLR/LMG/Johnson/LOFO are used elsewhere.

## 7.4 Suggested search keywords
- `tmax_member_hotcold_group_by_stage generation Hot17 Cold17 Total stage`
- `ECMWF S2S accumulated precipitation forecast step GRIB`
- `ERA5 geopotential conversion geopotential meters 9.80665`
- `LMG relative importance regression all orderings R2`
- `Johnson relative weights correlated predictors`
- `leave one factor out LOFO relative importance regression`

---

# 8. Code audit issue checklist

| Issue | Status | Evidence / comment |
|---|---|---|
| Code, figure labels, and terminology are consistent for Fig.9 Z500 absolute | Confirmed consistent | Fig.9 stores `z500_abs_*`, metadata `z500_type=absolute`, and plots absolute contours. |
| Same variable has conflicting definitions | Needs clarification | Z500 anomaly exists in Cell Y/Cell Z, while Fig.9 uses absolute Z500. Methods must separate diagnostics. |
| Units are consistent | Confirmed consistent for TP/Z500 in audited scripts | TP set to mm day^-1; Z500 to gpm. Need dataset metadata confirmation. |
| Standardized and raw coefficients are mixed | Not found in current window | No regression coefficient code found. |
| Bias/RMSE ensemble-mean/member-level mixing | Not found in current window | No Bias/RMSE implementation found. |
| Stage-I/Stage-II/Total terminology confusion | Potential inconsistency | File names mention Stage-II, but Fig.7/Fig.8/Fig.9 current formal grouping is Total-stage. Methods should state this clearly. |
| Hot17/Cold17 regrouped across different stages | Needs clarification | Current code reads Total groups per init. Other diagnostic scripts may still contain Stage-II naming and should be checked. |
| LMG/Johnson/LOFO called “contribution” | Not found in current window | Methods text should not discuss these as implemented here. |
| Partial fitted component interpreted causally | Not found in current window | No partial fitted component code found. |
| Legend/CSV/Methods naming consistency | Needs clarification | Cache fields use `Hot17_minus_Cold17`; plot labels use `Hot17 − Cold17`; both are sign-consistent but naming should be standardized in Methods. |
| p<0.05 spatial significance | Not found in current window | No scipy/stats/ttest/p-value implementation found. |
| Multi-init grouping per init | Confirmed consistent | Fig.9 loops over each `INIT_DATES` value and validates 17 members per init/group. |
| Hovmöller lat band and reference lines documented | Confirmed consistent | Fig.8 metadata records lat band and 114E/119E reference lines. |

---

# 9. Methods absent from current window but requiring separate audit

The following items were requested by the advisor but are **not implemented in the current spatial-propagation/Fig.7–Fig.9 window** based on text search and code inspection:

- Signed Bias.
- RMSE.
- Member-wise RMSE.
- RMSE of ensemble-mean forecast.
- Ensemble spread.
- Ensemble quantiles.
- Stage-mean error metrics.
- Multiple linear regression.
- Predictor set and response definition for regression.
- Standardized coefficients or raw-equivalent coefficients.
- Residualized predictors such as SHF residual, TCC residual, or SM residual.
- LMG relative importance.
- Johnson relative weights.
- LOFO.
- Correlation, R², slope and significance for regression.
- SM–SHF partial fitted component.
- Centered contribution and fitted component returned to Tmax space.
- Historical Tmax climatology, anomaly, KDE/probability density, event percentile, historical rank.
- Spatial p<0.05 significance mask/test.

These methods should be audited in the scripts/notebooks that generate Tmax evaluation tables, regression attribution tables, and historical-background diagnostics before they are described in Methods.
