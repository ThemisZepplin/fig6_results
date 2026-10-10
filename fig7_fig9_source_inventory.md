# Fig.7–Fig.9 source inventory for cache workflow

## Common cache paths
- Cache directory: `/data1/huangy/fig6/NC/v12/paper_figures/cache/`
- From-cache figure directory: `/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/`

## Fig.7 daily synoptic evolution
- Source script: `cell_fig7_stageII_daily_synoptic_paper.py`
- Cache file: `fig7_daily_synoptic_plot_cache.nc`
- Cache coordinates: `date`, `lat`, `lon`
- Variables to cache:
  - `tp_obs(date, lat, lon)`: MSWEP daily precipitation shading, mm day^-1.
  - `z500_obs(date, lat, lon)`: ERA5 absolute 500 hPa geopotential height contours, gpm.
  - `tp_hot(date, lat, lon)`: S2S Total Hot17 daily precipitation composite, mm day^-1.
  - `z500_hot(date, lat, lon)`: S2S Total Hot17 absolute Z500 composite, gpm.
  - `tp_cold(date, lat, lon)`: S2S Total Cold17 daily precipitation composite, mm day^-1.
  - `z500_cold(date, lat, lon)`: S2S Total Cold17 absolute Z500 composite, gpm.
  - `tp_diff(date, lat, lon)`: Hot17 minus Cold17 precipitation difference, mm day^-1.
  - `z500_diff(date, lat, lon)`: Hot17 minus Cold17 Z500 difference, gpm.
- TP type: BJT daily fields for 2023-06-18 to 2023-06-20.
- Z500 type: absolute for obs/hot/cold, difference for Hot17-Cold17.
- Grouping file: `/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv`
- Grouping stage: Total
- Main init: 2023-06-12
- Sign convention: Hot17_minus_Cold17 = Hot17 - Cold17

## Fig.8 Hovmöller propagation diagnosis
- Source script: `cell_hovmoller_tp_z500_obs_s2s_stageII.py`
- Cache file: `fig8_hovmoller_plot_cache.nc`
- Cache coordinates: `time`, `lon`
- Variables to cache:
  - `tp_obs_hov(time, lon)`: MSWEP Hovmöller precipitation, mm day^-1.
  - `z500_obs_hov(time, lon)`: ERA5 absolute Z500 Hovmöller contours, gpm.
  - `tp_hot_hov(time, lon)`: S2S Total Hot17 precipitation Hovmöller, mm day^-1.
  - `z500_hot_hov(time, lon)`: S2S Total Hot17 absolute Z500 Hovmöller, gpm.
  - `tp_cold_hov(time, lon)`: S2S Total Cold17 precipitation Hovmöller, mm day^-1.
  - `z500_cold_hov(time, lon)`: S2S Total Cold17 absolute Z500 Hovmöller, gpm.
  - `tp_diff_hov(time, lon)`: Hot17 minus Cold17 precipitation Hovmöller difference, mm day^-1.
  - `z500_diff_hov(time, lon)`: Hot17 minus Cold17 Z500 Hovmöller difference, gpm.
- TP type: BJT daily Hovmöller latitude-band average.
- Z500 type: absolute for obs/hot/cold, difference for Hot17-Cold17.
- Hovmöller lat band: 35–44N
- Longitude range: 90–135E
- Vertical reference lines: 114E, 119E
- Grouping file: `/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv`
- Grouping stage: Total
- Main init: 2023-06-12
- Analysis window: 2023-06-14 to 2023-06-24
- Sign convention: Hot17_minus_Cold17 = Hot17 - Cold17

## Fig.9 lead-time Total Hot/Cold composite maps
- Source script: `cell_leadtime_total_hotcold_composite_maps.py`
- Cache file: `fig9_leadtime_tp_z500_plot_cache.nc`
- Cache coordinates: `init`, `lat`, `lon`
- Variables to cache:
  - `tp_hot(init, lat, lon)`: Total Hot17 period-mean precipitation, mm day^-1.
  - `z500_abs_hot(init, lat, lon)`: Total Hot17 period-mean absolute Z500, gpm.
  - `tp_cold(init, lat, lon)`: Total Cold17 period-mean precipitation, mm day^-1.
  - `z500_abs_cold(init, lat, lon)`: Total Cold17 period-mean absolute Z500, gpm.
  - `tp_diff(init, lat, lon)`: Hot17 minus Cold17 period-mean precipitation difference, mm day^-1.
  - `z500_abs_diff(init, lat, lon)`: Hot17 minus Cold17 period-mean absolute Z500 difference, gpm.
  - `tp_obs_ref(lat, lon)`: MSWEP period-mean precipitation reference, mm day^-1.
  - `z500_abs_obs_ref(lat, lon)`: ERA5 period-mean absolute Z500 reference, gpm.
- TP type: 2023-06-14 to 2023-06-24 total-period mean.
- Z500 type: absolute geopotential height for hot/cold/reference, Hot17-Cold17 absolute-field difference for diff.
- Grouping file: `/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv`
- Grouping stage: Total
- Inits: 2023-06-01, 2023-06-05, 2023-06-08, 2023-06-12
- Sign convention: Hot17_minus_Cold17 = Hot17 - Cold17
