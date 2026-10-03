import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from experiments.CelebA.attribute_regression import stable_log_complement, regress, align_attributes


def test_log_complement_at_saturation():
    z=np.array([-1000.,-10.,0.,10.,1000.])
    y=stable_log_complement(z)
    assert np.isfinite(y).all()
    np.testing.assert_allclose(y[[0,2,4]],[np.log(2),0,np.log(2)-1000])
    middle=z[1:4]
    np.testing.assert_allclose(y[1:4],np.log(2-2/(1+np.exp(-middle))),atol=1e-11)
    with pytest.raises(ValueError):stable_log_complement([np.nan])


def test_joint_ols_partial_r_squared_matches_drop_one():
    rng=np.random.default_rng(114)
    x=rng.integers(0,2,size=(250,3));ids=np.repeat(np.arange(50),5)
    y=.1+x@np.array([1.,-.5,.2])+rng.normal(size=250)*.1
    table,metric=regress(y,x,['a','b','c'],ids)
    full=sm.OLS(y,sm.add_constant(x)).fit()
    np.testing.assert_allclose(table.coef,full.params[1:])
    for j in range(3):
        reduced=sm.OLS(y,sm.add_constant(np.delete(x,j,axis=1))).fit()
        np.testing.assert_allclose(table.partial_r2.iloc[j],(reduced.ssr-full.ssr)/reduced.ssr)
    assert metric['identities']==50
    assert (table.cluster_se>0).all()
    assert ((table.bh_qvalue>=0)&(table.bh_qvalue<=1)).all()


def test_annotations_align_by_filename(tmp_path):
    names=[f'a{i}' for i in range(40)]
    path=tmp_path/'attributes.txt'
    path.write_text('2\n'+' '.join(names)+'\nb.jpg '+' '.join(['1']*40)+'\na.jpg '+' '.join(['-1']*40)+'\n')
    a,n=align_attributes(path,pd.DataFrame({'source_id':['a.jpg','b.jpg']}))
    np.testing.assert_array_equal(a,np.array([[0]*40,[1]*40]));assert n==names
    with pytest.raises(ValueError):align_attributes(path,pd.DataFrame({'source_id':['missing.jpg']}))
