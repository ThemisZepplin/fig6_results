import numpy as np
import pandas as pd
import xarray as xr
import pytest
from ncvi_daily_exporter import load_pair, digest


def source_pair(tmp_path, monkeypatch, hindcast, pf_count=None):
    init='2023-06-08'
    refs=pd.date_range('2003-06-08',periods=20,freq=pd.DateOffset(years=1)) if hindcast else pd.DatetimeIndex([init])
    count=pf_count if pf_count is not None else (10 if hindcast else 50)
    steps=pd.to_timedelta([0,24,48],unit='h').to_numpy()
    sources={};spec={}
    for role,ids in [('cf',[0]),('pf',range(1,count+1))]:
        path=tmp_path/(role+'.grib');path.write_bytes(role.encode())
        values=np.zeros((len(refs),3,len(ids),1,1))
        for year in range(len(refs)):
            for j,member in enumerate(ids):values[year,:,j,0,0]=(year*100+member)*1e-6
        a=xr.DataArray(values,dims=['time','step','number','latitude','longitude'],
                       coords={'time':refs,'step':steps,'number':list(ids),'latitude':[40.],'longitude':[120.],'theta':320.},
                       attrs={'units':'K m**2 kg**-1 s**-1','GRIB_typeOfLevel':'theta','GRIB_shortName':'pv'})
        sources[str(path)]=a.to_dataset(name='pv')
        spec[role]={'path':str(path),'sha256':digest(path),'selected_reference_times':[str(t) for t in refs]}
    monkeypatch.setattr(xr,'open_dataset',lambda path,**kwargs:sources[str(path)].copy(deep=True))
    return init,spec,steps


@pytest.mark.parametrize('hindcast,shape',[(False,(3,51)),(True,(3,220))])
def test_real_member_and_time_structure(tmp_path,monkeypatch,hindcast,shape):
    init,spec,steps=source_pair(tmp_path,monkeypatch,hindcast)
    values,actual_steps,audit=load_pair(spec,init,hindcast)
    assert values.shape==shape and np.array_equal(actual_steps,steps)
    if hindcast:
        expected=np.array([year*100+member for year in range(20) for member in range(11)])
        np.testing.assert_allclose(values[0],expected,atol=1e-12)
        assert len(audit[0]['reference_times'])>0
    else:np.testing.assert_allclose(values[0],np.arange(51),atol=1e-12)


@pytest.mark.parametrize('hindcast,count',[(True,50),(False,10)])
def test_wrong_pf_structure_rejected(tmp_path,monkeypatch,hindcast,count):
    init,spec,_=source_pair(tmp_path,monkeypatch,hindcast,count)
    with pytest.raises(ValueError,match='PF IDs'):load_pair(spec,init,hindcast)
