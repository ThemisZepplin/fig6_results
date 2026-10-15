# 四起报 Fig.6 与 Fig.7 排版更新：Jupyter 运行说明

## 图号与范围

新正文 Fig.6 = 四个起报（6月1/5/8/12日）的6月18–20日平均空间图。
每个起报仍按6月14–24日 Total Tmax 分组，17名 Hot/17名 Cold；不重新排名。
Fig.7 = 原有 June 8 / June 12 七面板 Hovmöller，仅调整标题与日期。
原逐日 Fig.6/S6 作为 SI 候选保留；原14–24日空间图（此前对话 Fig.9，
会前稿 Fig.8）从正文移除，保留文件。最终早期窗口 MLR Fig.8 不在本轮修改。

本分支新建于最新 dev/jgr-major-revision；只选择性复制 PR13 所需脚本，
没有合并 PR13，没有修改旧分支。优先使用本文件列出的新入口，避免旧入口
plot_fig6_from_cache 在旧统一模块中仍代表逐日图的歧义。

## 0. 获取代码

将本 PR 的全部 .py 文件下载到服务器同一个目录，或检出本功能分支。
下列 CODE 必须改成**服务器上实际保存这些代码的绝对目录**，不是数据输出目录。
不要直接使用 Windows E: 路径；不需要把该目录加进 sys.path。

```python
from pathlib import Path
import runpy

CODE = Path("/请替换为服务器上的本次代码目录")
ROOT = Path("/data1/huangy/fig6/NC/v12/paper_figures")

required_scripts = [
    "cell_leadtime_total_hotcold_composite_maps.py",
    "cell_fig6_interruption_totalgroup_export.py",
    "paper_fig6_interruption_from_cache.py",
    "paper_fig8_from_cache.py",
    "paper_fig67_revision_from_cache.py",
]
missing = [CODE / name for name in required_scripts if not (CODE / name).is_file()]
if missing:
    raise FileNotFoundError("缺少代码文件：\n" + "\n".join(map(str, missing)))
```

## 1. 新 Fig.6 首次导出：一次运行同时生成 cache 和独立显著性

先检查是否已有本次指定的新文件。旧14–24日 nc、逐日 nc 及其 p 值不替代它们。
已存在的其他18–20日文件可另行核验，但不能只改文件名当成本次新产品。

```python
export_ns = runpy.run_path(str(CODE / "cell_fig6_interruption_totalgroup_export.py"))
products = export_ns["_fig6_product_paths"](ROOT)
missing_products = [p for p in products.values() if not p.is_file()]
for name, path in products.items():
    print(name, "存在" if path.is_file() else "缺失", path)

if missing_products:
    outputs = export_ns["export_fig6_interruption_cache_and_significance"](
        root=ROOT,
        source_path=CODE / "cell_leadtime_total_hotcold_composite_maps.py",
    )
    print(outputs)
else:
    print("五个新产品均存在，可直接进行严格校验和重绘。")
```

如果仅存在部分新产品，函数会明确报已存在文件，避免自动覆盖。确认要重新生成
**这一组新 Fig.6 产品**时，在上面的导出调用加 overwrite=True；旧产品不受影响。
导出过程中读取原始 S2S/ERA5/MSWEP 和 v9 grouping CSV，耗时属正常。
每个起报只加载一次成员场，同时导出 TP 与 Z500 的独立统计。

新的五个文件：

- cache/fig6_interruption_0618_0620_totalgroup_fourinit_plot_cache.nc
- cache/fig6_interruption_0618_0620_totalgroup_fourinit_cache_qc.csv
- significance/fig6_interruption_0618_0620_totalgroup_fourinit_significance_tp.nc
- significance/fig6_interruption_0618_0620_totalgroup_fourinit_significance_z500.nc
- significance/fig6_interruption_0618_0620_totalgroup_fourinit_significance_qc.csv

显著性以每个成员的三日平均场为样本；17 vs 17，逐起报、逐变量计算。
局地双侧 Welch p<0.05 用于图面，BH-FDR 仅保存在文件里用于补充核查。
不平均逐日 p 值、显著性掩膜，不沿用14–24日检验。

## 2. 新正文 Fig.6 重绘：后续仅运行这里即可

```python
renderers = runpy.run_path(str(CODE / "paper_fig67_revision_from_cache.py"))
fig6_paths = renderers["plot_fig6_from_cache"]()
print("新正文 Fig.6:", fig6_paths)
```

若使用自定义 ROOT，明确传入 cache_path、tp_significance_path 和 out_dir。
此绘图入口只读 cache NetCDF 和 TP significance NetCDF，不读取原始数据和分组表。
Z500 统计独立保存，但绘图无需读取它；不恢复高度场显著性粗线叠加。

输出：
final_from_cache/jgr_6p5in/fig6_interruption_0618_0620_totalgroup_fourinit.png/.pdf

布局沿用确认过的6.5×6.5英寸、四行×三列和观测(m)参考；
左侧起报文字 x=0.085，色标 y=0.160/0.085。色阶由本次四起报共同决定，
刻度从对应缓存色阶生成，不硬套旧11天平均幅度。
标签只能来自实际等值线级别，非 inline，不截断轮廓；无法安全标注时记录日志。
真实新数据的字号、标签、点状显著性和色阶对比仍需服务器输出后目视确认。

## 3. Fig.7：只读既有 Hovmöller cache 重绘

```python
inputs = {
    "cache_0608": ROOT / "cache/fig8_hovmoller_init2023-06-08_plot_cache.nc",
    "cache_0612": ROOT / "cache/fig8_hovmoller_plot_cache.nc",
    "tp_significance_0608": ROOT / "significance/fig8_init2023-06-08_hovmoller_significance_tp.nc",
    "tp_significance_0612": ROOT / "significance/fig8_hovmoller_significance_tp.nc",
}
missing = [p for p in inputs.values() if not p.is_file()]
if missing:
    raise FileNotFoundError("缺少 Hovmöller 输入：\n" + "\n".join(map(str, missing)))

renderers = runpy.run_path(str(CODE / "paper_fig67_revision_from_cache.py"))
fig7_paths = renderers["plot_fig7_from_cache"](
    **{name: str(path) for name, path in inputs.items()},
    out_dir=str(ROOT / "final_from_cache/jgr_6p5in"),
)
print("Fig.7:", fig7_paths)
```

保留(a)居中观测，(b,c) Hot，(d,e) Cold，(f,g) 差值，左 June 8、右 June 12；
日期显示6-14…6-24。标题和序号合并放框上方，11pt加粗；七图轴框等宽等高。
为保留旧图，新输出：
fig7_hovmoller_total_init2023-06-08_2023-06-12_revised.png/.pdf

只有确实缺文件才加载对应科学脚本；例如 June 8 缺文件时，在**独立命名空间**调用：

```python
science_hov = runpy.run_path(str(CODE / "cell_hovmoller_tp_z500_obs_s2s_stageII.py"))
# 仅在对应文件缺失时执行：
# science_hov["export_fig8_20230608_plot_cache"]()
# science_hov["export_fig8_20230608_hovmoller_significance"]()
```

## 4. 逐日图迁入 SI 候选目录

无需重算逐日数据，也无需重复绘制才能继续新 Fig.6。
如果要另存 SI 候选，可运行（需保留 paper_fig7_from_cache.py）：

```python
renderers = runpy.run_path(str(CODE / "paper_fig67_revision_from_cache.py"))
for init in ("2023-06-08", "2023-06-12"):
    print(init, renderers["plot_daily_si_candidate"](initialization=init))
```

输出目录 final_from_cache/si_daily_candidates/；保留原日图文件名以溯源，
不占用新 Fig.6 输出路径。最终 Fig.S? 图号待 SI 总体编排。
旧14–24日空间图、其 cache/显著性和服务器上已确认的最终图片均保留；
如需另存档案，使用 plot_total_mean_archive_from_cache()，输出至 archive_total_0614_0624。

## 5. 核验与边界

- 查看 cache QC 的单位、完整性、TP 最大值位置及 NCHN 指标。
- significance QC 应有8行（4起报×TP/Z500），expected_day_count=3，样本17/17。
- from-cache 强制核验目标日期、平均日数、分组窗口、分组成员、四起报、
  原生坐标、TP/Z500单位与差值、p值范围及掩膜一致性。
- 新目标平均时段是本次明确授权的计算差异；原始读取、日界、累积量拆分、
  单位转换、Total 分组定义和 Welch/BH 统计核保持旧版。
- CI 是合成数据与渲染机制验证，不代表真实服务器出图验收。
