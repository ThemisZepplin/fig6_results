"""Revised Fig.3/4/9/10 synthetic QC. No real HPC results are asserted."""
import json
from pathlib import Path
import numpy as np
import pytest
import matplotlib
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
    built=plot.make_figures(data,plan,"scatter")+plot.make_figures(data,plan,"importance")
    assert len(built)==4
    output=plot.save_figures(built,inp,out)
    assert len(output)==4
    assert set(output.figure)=={"Fig3","Fig4","Fig9","Fig10"}
    assert set(output.version)=={"residualized"}
    assert (output.status=="PASS").all()
    assert oldhash==hashlib.sha256(old.read_bytes()).hexdigest()
    assert all_tables=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/"tables").glob("*.csv")}
    assert (out/"qc/figure_lofo_deltaR2_v1_qc.json").is_file()
    for figure in ("Fig4","Fig10"):
        pdf=next(out.glob(f"figures/{figure}_*.pdf"))
        assert pdf.stat().st_size>1000
    with pytest.raises(FileExistsError):
        plot.save_figures(plot.make_figures(data,plan,"scatter")+plot.make_figures(data,plan,"importance"),inp,out)

def test_reference_ranking_violation_rejected(fixture):
    _,_,data,_=fixture
    copied=dict(data)
    copied["importance"]=data["importance"].copy()
    mask=(copied["importance"].init_date=="2023-06-12")&(copied["importance"].window=="Total")&(copied["importance"].version=="residualized")&(copied["importance"].method=="LMG")
    copied["importance"].loc[mask,"importance_pct"]=[1,2,3,4,5,85]
    with pytest.raises(ValueError,match="12 June residualized LMG order mismatch"):
        plot.figure_plan(copied)
