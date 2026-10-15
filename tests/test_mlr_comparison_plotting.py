"""Revised Fig.3/4/9/10 synthetic QC. No real HPC results are asserted."""
import json
from pathlib import Path
import numpy as np
import pytest
import matplotlib
from matplotlib import patheffects
matplotlib.use("Agg")
import mlr_comparison_core as core
import mlr_comparison_plotting as plot
from test_mlr_comparison import synthetic_inputs

@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp("mlr_deltaR2")
    inp,out=root/"input",root/"model_output"
    raw,nc=synthetic_inputs()
    groups,ra=core.prepare_pairs(raw,nc)
    models=core.fit_pairs(groups)
    permutations=core.lmg_permutations()
    imp,qc=core.importance(models,permutations)
    results=core.assemble_results(groups,models,imp,qc,ra)
    # Only override plotting reference LMG percentages in synthetic data.
    for window,order in plot.IMPORTANCE_REFERENCE_ORDER.items():
        for predictor,pct in zip(order,[35,25,18,12,7,3]):
            mask=(results["importance"].init_date.eq("2023-06-12") &
                  results["importance"].window.eq(window) &
                  results["importance"].version.eq("residualized") &
                  results["importance"].method.eq("LMG") &
                  results["importance"].predictor.eq(predictor))
            results["importance"].loc[mask,"importance_pct"]=pct
    hist=core.compare_history(results,{})
    core.save_results(results,hist,inp,out,[{"synthetic_only":True}],permutations)
    data,_=plot.load_results(out)
    return inp,out,data,plot.figure_plan(data)

def test_only_residualized_and_metric_not_normalized(fixture):
    _,_,data,plan=fixture
    assert plot.FIGURE_VERSION=="residualized"
    assert set(plan["bars"])=={"LMG","Johnson","LOFO"}
    assert plan["bars"]["LOFO"]<1
    items=plot.make_figures(data,plan,"importance")
    assert len(items)==2
    for item in items:
        assert item["qc"]["version"]=="residualized"
        axes=item["fig"].axes
        assert len(axes)==12
        for j,ax in enumerate(axes):
            method=core.METHODS[j//4]
            assert len(ax.get_xticks())>=2
            assert len(ax.get_xticklabels())>=2
            init=core.INITS[j%4]
            window=item["qc"]["window"]
            imp=data["importance"]
            group=imp[(imp.method==method)&(imp.window==window)&
                      (imp.init_date==init)&(imp.version=="residualized")]
            group=group.set_index("position").loc[plot.importance_order(data,window)]
            expected=group.raw_value if method=="LOFO" else group.importance_pct
            np.testing.assert_allclose([p.get_width() for p in ax.patches],expected,rtol=0,atol=1e-10)
            labels=[a.get_text() for a in ax.texts]
            assert len(labels)==6
            assert all(not label.endswith("%") for label in labels) if method=="LOFO" else all(label.endswith("%") for label in labels)
        plot.plt.close(item["fig"])

def test_scatter_axes_and_stats(fixture):
    _,_,data,plan=fixture
    items=plot.make_figures(data,plan,"scatter")
    assert len(items)==2
    for item in items:
        for ax in item["fig"].axes:
            assert ax.get_xlabel()=="ECMWF Tmax (°C)"
            assert ax.get_ylabel()=="MLR predicted Tmax (°C)"
            assert "slope =" in ax.texts[0].get_text()
            assert "R² =" in ax.texts[0].get_text()
            assert "r =" not in ax.texts[0].get_text()
        plot.plt.close(item["fig"])

def test_original_lofo_audit_export(fixture):
    _,out,data,_=fixture
    filename=out/"tables/table_mlr_lofo_original_delta_r2_residualized.csv"
    assert filename.is_file()
    import pandas as pd
    q=pd.read_csv(filename)
    assert len(q)==4*2*6
    assert (q.n==51).all()
    np.testing.assert_allclose(q.lofo_delta_r2,q.R2_full-q.R2_without_factor,rtol=0,atol=1e-8)
    assert (q.lofo_delta_r2>=-1e-8).all()
    legacy=data["importance"].query("method=='LOFO' and version=='residualized'")
    assert len(legacy)==len(q)

def test_export_four_only_and_keep_original_numerical_tables(fixture):
    inp,out,data,plan=fixture
    import hashlib
    old=out/"figures/old_raw_figure.png"
    old.parent.mkdir(parents=True,exist_ok=True)
    old.write_bytes(b"old figure preserved")
    oldhash=hashlib.sha256(old.read_bytes()).hexdigest()
    all_tables={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/"tables").glob("*.csv")}
    built=plot.make_figures(data,plan,"importance")
    assert len(built)==2
    output=plot.save_figures(built,inp,out)
    assert len(output)==2
    assert set(output.figure)=={"Fig4","Fig10"}
    assert set(output.version)=={"residualized"}
    assert (output.status=="PASS").all()
    assert oldhash==hashlib.sha256(old.read_bytes()).hexdigest()
    assert all_tables=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/"tables").glob("*.csv")}
    assert (out/"qc/figure_lofo_deltaR2_v4_qc.json").is_file()
    for figure in ("Fig4","Fig10"):
        pdf=next(out.glob(f"figures/{figure}_*.pdf"))
        assert pdf.stat().st_size>1000
    with pytest.raises(FileExistsError):
        plot.save_figures(plot.make_figures(data,plan,"importance"),inp,out)

def test_reference_ranking_violation_rejected(fixture):
    _,_,data,_=fixture
    copied=dict(data)
    copied["importance"]=data["importance"].copy()
    mask=(copied["importance"].init_date=="2023-06-12")&(copied["importance"].window=="Total")&(copied["importance"].version=="residualized")&(copied["importance"].method=="LMG")
    copied["importance"].loc[mask,"importance_pct"]=[1,2,3,4,5,85]
    with pytest.raises(ValueError,match="12 June residualized LMG order mismatch"):
        plot.figure_plan(copied)


def test_legacy_manifest_with_no_extra_lofo_csv_is_supported(fixture, tmp_path):
    """Reproduce uploaded v13 manifest: 11 table SHA entries, no derived LOFO CSV."""
    import shutil
    inp, out, _, _ = fixture
    legacy = tmp_path / "legacy_v13"
    shutil.copytree(out, legacy)
    target = legacy / "tables/table_mlr_lofo_original_delta_r2_residualized.csv"
    target.unlink()
    marker = legacy / "qc/comparison_manifest.json"
    manifest = json.loads(marker.read_text())
    manifest["table_sha256"].pop(target.name)
    marker.write_text(json.dumps(manifest))
    data, audit = plot.load_results(legacy)
    assert audit["lofo_plot_input_audit_mode"] == "LEGACY_HASHED_IMPORTANCE_AND_REDUCED_R2"
    assert len(data["importance"]) == 288
    assert plot.figure_plan(data)["bars"]["LOFO"] < 1


def test_legacy_lofo_source_tampering_is_rejected(fixture, tmp_path):
    import shutil
    _, out, _, _ = fixture
    legacy = tmp_path / "legacy_corrupted"
    shutil.copytree(out, legacy)
    target = legacy / "tables/table_mlr_lofo_original_delta_r2_residualized.csv"
    target.unlink()
    marker = legacy / "qc/comparison_manifest.json"
    manifest = json.loads(marker.read_text())
    manifest["table_sha256"].pop(target.name)
    marker.write_text(json.dumps(manifest))
    source = legacy / "tables/table_mlr_importance_long.csv"
    txt = source.read_text()
    source.write_text(txt.replace("LOFO", "LOOO", 1))
    with pytest.raises(ValueError, match="Table changed"):
        plot.load_results(legacy)


def test_revised_inside_labels_never_erase_bar_tips(fixture):
    _, _, data, plan = fixture
    for item in plot.make_figures(data, plan, "importance"):
        qc = item["qc"]
        assert qc["label_placement_status"] == "PASS", qc["label_placement_failures"]
        assert qc["label_placement_count"] == 72
        assert qc["label_placement_inside"] > 0
        assert qc["label_placement_outside"] > 0
        assert all(record["valid"] for record in qc["label_placement_records"])
        assert item["qc"]["basename"].endswith("_lofo_deltaR2_v4")
        for ax in item["fig"].axes:
            assert all(label.get_bbox_patch() is None for label in ax.texts)
            for label in ax.texts:
                effects = label.get_path_effects()
                if label._mlr_placement == "inside":
                    assert len(effects) == 1
                    assert isinstance(effects[0], patheffects.withStroke)
                    assert effects[0]._gc["linewidth"] == pytest.approx(0.6)
                    assert effects[0]._gc["foreground"] == "white"
                else:
                    assert label._mlr_placement == "outside"
                    assert effects == []
        plot.plt.close(item["fig"])


def test_positive_tiny_lofo_not_displayed_as_zero(fixture):
    _, _, data, plan = fixture
    copied = dict(data)
    copied["importance"] = copied["importance"].copy()
    sub = ((copied["importance"].init_date == core.INITS[0]) &
           (copied["importance"].window == "Total") &
           (copied["importance"].version == "residualized") &
           (copied["importance"].method == "LOFO"))
    index = copied["importance"].index[sub][0]
    copied["importance"].loc[index, "raw_value"] = 0.000095
    with plot.plt.rc_context(plot.STYLE):
        fig, axes, labels = plot.importance_figure(copied, plan, "Total", "residualized")
    assert "<0.001" in [t.get_text() for ax in axes[8:] for t in ax.texts]
    plot.plt.close(fig)


def test_halo_preserves_native_pdf_text_extraction(tmp_path):
    """A stroke-only PDF loses text; layered halo must retain native glyph text."""
    import fitz
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    t = plot.ExtractableHaloText(.5, .5, "42.2%", fontsize=10, color="black",
                                 ha="center", va="center")
    ax.add_artist(t)
    t.set_path_effects([patheffects.withStroke(
        linewidth=plot.INSIDE_LABEL_HALO_PT, foreground="white")])
    pdf = tmp_path / "label-halo.pdf"
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(pdf)
    with fitz.open(pdf) as doc:
        assert doc[0].get_text().count("42.2%") == 1
        assert not doc[0].get_images()
    assert t.get_bbox_patch() is None
    plt.close(fig)

