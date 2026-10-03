"""Null target and fresh-fit/final-evaluation integration contracts."""
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.CelebA import null_experiment as null


def test_five_families_use_fixed_accepted_settings():
    tasks = null.tasks_for({'repeats':5})
    assert len(tasks) == 25
    assert len({(t['representation'],t['source'],t['repeat']) for t in tasks}) == 25
    assert not any(t['representation']=='feature' and t['source']=='real' for t in tasks)
    assert all(t['loss']=='js' and t['output_alpha']==(.5 if t['representation']=='feature' else 2) for t in tasks)


def test_null_truth_and_population_brier_excess_identity():
    exact = null.null_metrics(np.ones(7),np.ones(3))
    assert exact['rdr_mse_one']==0 and exact['excess_brier_true']==0
    assert exact['fraction_within_0.05']==1
    biased = null.null_metrics(np.full(7,1.2),np.full(3,1.2))
    assert biased['rdr_mse_one']==pytest.approx(.04)
    assert biased['excess_brier_true']==pytest.approx(.01)
    for bad in [[],[np.nan],[2.1]]:
        with pytest.raises(ValueError): null.null_metrics(bad,[1.])


@pytest.mark.parametrize('level',['feature','pixel'])
def test_fresh_null_fit_uses_separate_final_roles_and_preserves_learned_scores(tmp_path,monkeypatch,level):
    from experiments.CelebA import null_data
    torch.set_num_threads(1)
    cfg={'seed':9,'bins':20,'ci_alpha':.05,'repeats':1,
         'feature':{'input_dimension':3,'hidden_dimension':4,'batch_size':4,'max_epochs':1,'min_epochs':1,'patience':1},
         'pixel':{'ndf':2,'batch_size':4,'max_epochs':1,'min_epochs':1,'patience':1,'bn_freeze_epoch':1}}
    task=next(t for t in null.tasks_for(cfg) if t['representation']==level)
    protocol={'config':cfg,'tasks':[task]}
    null.write(tmp_path/'protocol.json',{})
    monkeypatch.setattr(null,'locked',lambda out:protocol)
    calls=[]
    rng=np.random.default_rng(2)
    def load(output,representation,source,roles):
        calls.append(tuple(roles))
        arrays={f'{role}_{side}':(rng.normal(size=(8,3)).astype('float32') if level=='feature'
                                  else rng.integers(0,256,(8,3,64,64),dtype='uint8'))
                for role in roles for side in ('p','q')}
        return {key:null_data.IndexedRows(value,np.arange(len(value)-1,-1,-1)) for key,value in arrays.items()}
    monkeypatch.setattr(null_data,'load_null_arrays',load)
    null.fit(tmp_path,0,'cpu')
    assert calls==[null.DEVELOPMENT,null.FINAL]
    assessment=null.directory(tmp_path,task,'evaluation')
    row=null.read(assessment/'metrics.json')
    with np.load(assessment/'predictions.npz') as saved:
        p,q=saved['test_evaluation_p'],saved['test_evaluation_q']
    assert row['rdr_mse_one']==pytest.approx(null.null_metrics(p,q)['rdr_mse_one'])
    assert not (np.all(p==1) and np.all(q==1))
    assert row['empirical_brier_excess']==pytest.approx(row['brier']-.25)
    assert row['null_truth_rdr']==1
    null.fit(tmp_path,0,'cpu')
    assert len(calls)==2
    (assessment/'predictions.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'): null.fit(tmp_path,0,'cpu')
