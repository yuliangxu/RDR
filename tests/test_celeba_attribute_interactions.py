import importlib.util
from pathlib import Path
import numpy as np
import pytest
ENV=Path('/cwork/yx306/RDR/CelebA/environments/interpret_core_0_7_8')
if not ENV.exists() and importlib.util.find_spec('interpret') is None:
    pytest.skip('Requires isolated InterpretML environment',allow_module_level=True)
from experiments.CelebA.attribute_interactions import identity_split, fit_surrogate, score, PARAMS, GROUPS


def test_split_is_identity_disjoint_and_repeatable():
    identities=np.repeat(np.arange(100),np.arange(100)%7+1)
    roles=identity_split(identities)
    np.testing.assert_array_equal(roles,identity_split(identities))
    for value in np.unique(identities):assert len(np.unique(roles[identities==value]))==1
    counts={r:len(np.unique(identities[roles==r])) for r in ('train','val','test')}
    assert counts==dict(train=60,val=20,test=20)


def test_surrogate_cannot_use_outer_test_targets():
    rng=np.random.default_rng(20);x=rng.integers(0,2,size=(600,3)).astype(float)
    y=x[:,0]+2*x[:,1]*x[:,2]+rng.normal(0,.1,len(x));roles=identity_split(np.repeat(np.arange(120),5))
    params={**PARAMS,'max_rounds':30,'early_stopping_rounds':5}
    m,a,b=fit_surrogate(x,y,roles,['a','b','c'],1,params)
    altered=y.copy();altered[roles=='test']+=1000
    n,c,d=fit_surrogate(x,altered,roles,['a','b','c'],1,params)
    np.testing.assert_array_equal(m.predict(x),n.predict(x));assert a==c and b==d
    assert a==pytest.approx(y[roles=='train'].mean())


def test_delta_r2_equals_variance_normalized_error_increase():
    y=np.array([0.,1.,2.,3.]);full=score(y,y+.1);reduced=score(y,y+.5)
    assert full['r_squared']-reduced['r_squared']==pytest.approx((reduced['mse']-full['mse'])/y.var())
    assert reduced['r_squared']-full['r_squared']<0
    assert len(GROUPS)==7
