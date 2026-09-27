"""Scientific contracts for the retained final Agent 3 workflow."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from CELEBA_rdr import estimate_midpoint_hellinger, original_hellinger_midpoint_loss
from CELEBA_final_test import prepare_rdr_bootstrap_components, rdr_bootstrap_row
from CELEBA_selection import FeatureMoments, fid_from_moments
from CELEBA_plots import select_nearest_in_bin, BIN_SPECS
from CELEBA_data import load_yaml
from CELEBA_workflow import ManifestImages, model_and_optimizer, image_shard


def test_null_target_and_equal_mixture_maximum():
    a=np.ones(32)
    assert estimate_midpoint_hellinger(a,a,1e-6)['variational_lower_bound']==0
    result=estimate_midpoint_hellinger(np.full(32,2.),np.zeros(32),1e-12)
    assert abs(result['variational_lower_bound']-(1-1/np.sqrt(2)))<1e-6


def test_training_and_report_objective_agree():
    p=torch.tensor([.2,.8,1.9],dtype=torch.float64)
    q=torch.tensor([.1,.5,1.3],dtype=torch.float64)
    expected=-estimate_midpoint_hellinger(p.numpy(),q.numpy(),1e-6)['variational_lower_bound']
    assert abs(original_hellinger_midpoint_loss(torch.nn.Identity(),p,q,1e-6).item()-expected)<1e-14


def test_pooled_test_is_weighted_mean_of_component_objectives():
    rng=np.random.default_rng(42)
    p=rng.uniform(.01,1.99,100);q=rng.uniform(.01,1.99,100)
    h=lambda a,b:estimate_midpoint_hellinger(a,b,1e-6)['variational_lower_bound']
    assert np.isclose(h(p,q),.37*h(p[:37],q[:37])+.63*h(p[37:],q[37:]),atol=1e-14)


def test_bootstrap_preserves_pairing_and_shared_real_rows():
    p=np.linspace(.1,1.9,40);q=p[::-1].copy()
    scores={b:{r:{'p':p,'q':q} for r in ['feature','pixel']} for b in ['lower','upper']}
    components=prepare_rdr_bootstrap_components(scores,1e-6)
    row=rdr_bootstrap_row(components,123,0)
    assert row['pixel_minus_feature_real_vs_lower']==0
    assert row['pixel_minus_feature_real_vs_upper']==0
    assert row==rdr_bootstrap_row(components,123,0)


def test_fid_known_gaussians():
    p=FeatureMoments(np.zeros(3),np.eye(3),100)
    q=FeatureMoments(np.ones(3),4*np.eye(3),100)
    assert np.isclose(fid_from_moments(p,q).fid,6.)


def test_score_selection_respects_bins_and_targets():
    frame=pd.DataFrame({'source_id':['a','b','c','d'],'rdr':[0.,.49,.5,1.]})
    chosen,n=select_nearest_in_bin(frame,'rdr',BIN_SPECS[0],1,123)
    assert n==2 and chosen.source_id.tolist()==['a']
    chosen,n=select_nearest_in_bin(frame,'rdr',BIN_SPECS[1],1,123)
    assert n==2 and chosen.source_id.tolist()==['d']


def test_uint8_input_scales_and_fresh_shard_resolution(tmp_path):
    original=tmp_path/'source.pt';torch.save({'images':torch.full((1,3,64,64),255,dtype=torch.uint8)},original)
    frame=pd.DataFrame([dict(source='stylegan2_lower',tensor_path=str(original),idx_in_shard=0)])
    assert torch.all(ManifestImages(frame,tmp_path)[0]==1)
    with pytest.raises(FileNotFoundError):ManifestImages(frame,tmp_path,fresh=True)[0]
    fresh=image_shard(tmp_path,original);fresh.parent.mkdir();torch.save({'images':torch.zeros((1,3,64,64),dtype=torch.uint8)},fresh)
    assert torch.all(ManifestImages(frame,tmp_path,fresh=True)[0]==-1)
    assert torch.all(ManifestImages(frame,tmp_path,fresh=True,zero_one=True)[0]==0)
    dataset=ManifestImages(frame,tmp_path,fresh=True)
    before=dataset[0].clone()
    dataset.preload()
    assert torch.equal(before,dataset[0])


def test_final_models_keep_exact_architecture_and_no_constant_selection():
    root=Path(__file__).resolve().parents[1]/'configs'
    cfg=load_yaml(root/'agent3_real_generator_rdr.yaml')
    assert cfg['selection']['test']['sample_size_per_distribution']==18000
    for level in ['feature','pixel']:
        assert cfg[level+'_rdr']['constant_ratio_validation_baseline'] is False
        model,_=model_and_optimizer(level,cfg[level+'_rdr'])
        model.eval()
        x=torch.zeros((2,2048)) if level=='feature' else torch.zeros((2,3,64,64))
        with torch.no_grad():r=model(x)
        assert torch.isfinite(r).all() and r.min()>=0 and r.max()<=2
