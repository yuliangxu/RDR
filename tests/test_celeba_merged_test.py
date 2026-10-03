"""Pooled test rows and global image rankings must match the registered inputs."""
import numpy as np
import pytest
from experiments.CelebA.merged_test import merge_scores, diagnostics, ranked_indices, JoinedImages, montage
from utils.calibration import bin_scores, calibrate_counts


def test_merge_preserves_role_order_and_rejects_wrong_counts():
    predictions={'test_calibration_p':np.array([.1,.2]),'test_evaluation_p':np.array([.3]),
                 'test_calibration_q':np.array([1.1]),'test_evaluation_q':np.array([1.2,1.3])}
    counts={'test_calibration':{'p':2,'q':1},'test_evaluation':{'p':1,'q':2}}
    pooled=merge_scores(predictions,counts)
    np.testing.assert_array_equal(pooled['p'],[.1,.2,.3])
    np.testing.assert_array_equal(pooled['q'],[1.1,1.2,1.3])
    counts['test_calibration']['p']=3
    with pytest.raises(ValueError,match='count'):merge_scores(predictions,counts)


def test_shared_test_counts_ci_and_source_balancing():
    p=np.array([0.,.04,.12,1.,2.]);q=np.array([0.,1.1,1.9])
    summary,cells=diagnostics({'p':p,'q':q},null=True)
    edges=np.linspace(0,2,21)
    kp=np.bincount(bin_scores(p,edges),minlength=20)
    kq=np.bincount(bin_scores(q,edges),minlength=20)
    reference=calibrate_counts(kp,kq,len(p),len(q),.05)
    assert summary['n_test_p']==5 and summary['n_test_q']==3
    assert summary['supported_mass']==pytest.approx(1)
    assert summary['brier']==pytest.approx(.5*np.mean((p/2-1)**2)+.5*np.mean((q/2)**2))
    assert summary['rdr_mse_one']==pytest.approx(.5*np.mean((p-1)**2)+.5*np.mean((q-1)**2))
    np.testing.assert_array_equal([c['test_count_p'] for c in cells],kp)
    np.testing.assert_allclose([c['c2_lower'] for c in cells],reference['c2_lower'])
    np.testing.assert_allclose([c['c2_upper'] for c in cells],reference['c2_upper'])
    assert all('cal_count_p' not in c and 'eval_count_p' not in c for c in cells)
    assert cells[5]['calibrated_rdr'] is None
    assert cells[5]['c2_lower']==0 and cells[5]['c2_upper']==2


@pytest.mark.parametrize('target',[0,1,2])
def test_global_ranking_not_restricted_to_old_ranges(target):
    scores=np.array([.7,.8,.9,.99,1.,1.01,1.1,1.2,1.3])
    selected=ranked_indices(scores,target,3,19)
    np.testing.assert_allclose(np.sort(abs(scores[selected]-target)),np.sort(abs(scores-target))[:3])
    assert len(set(selected))==3
    np.testing.assert_array_equal(selected,ranked_indices(scores,target,3,19))


def test_ties_are_reproducible_and_images_follow_merged_row_order():
    scores=np.full(100,2.)
    chosen=ranked_indices(scores,2,40,5)
    assert len(set(chosen))==40
    np.testing.assert_array_equal(chosen,ranked_indices(scores,2,40,5))
    pieces=[np.full((2,3,64,64),17,dtype=np.uint8),np.full((1,3,64,64),93,dtype=np.uint8)]
    images=JoinedImages(pieces)
    picture=montage(images,[2,0],columns=2,capacity=2)
    assert np.all(picture[:64,:64]==93)
    assert np.all(picture[:64,65:]==17)
    assert np.all(picture[:,64]==255)
    with pytest.raises(ValueError):images[3]
