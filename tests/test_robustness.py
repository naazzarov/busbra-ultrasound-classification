import numpy as np
import pytest
from busbra.robustness import corrupt_training_labels,CriterionAblationSelector,class_extreme_neighbor_dependency,experiments
from busbra.selectors import GBFRSSelector,ball_dependency


def test_noise_flips_whole_training_groups_and_is_reproducible():
    groups=np.repeat(np.arange(100),2);y=np.repeat([0]*60+[1]*40,2)
    corrupted,audit=corrupt_training_labels(y,groups,fold=1,rate=.2)
    assert np.array_equal(corrupted,corrupt_training_labels(y,groups,1,.2)[0])
    assert audit['flipped_groups']==20 and audit['flipped_rows']==40
    assert np.all(corrupted[::2]==corrupted[1::2])
    assert np.array_equal(y,np.repeat([0]*60+[1]*40,2))
    assert np.array_equal(corrupt_training_labels(y,groups,1,0)[0],y)
    with pytest.raises(ValueError):corrupt_training_labels(y,groups,1,.5)


def test_controlled_dependency_changes_only_the_criterion():
    centers=np.array([[0.],[.2],[.9]]);labels=np.array([0,0,1]);weights=np.array([.6,.3,.1])
    assert ball_dependency(centers,labels,weights,[0],1)==pytest.approx(.82)
    assert class_extreme_neighbor_dependency(centers,labels,weights,[0],1)==pytest.approx(.51)
    rng=np.random.RandomState(10)
    X=np.r_[rng.normal(.1,.01,(30,5)),rng.normal(.8,.01,(30,5))];y=np.repeat([0,1],30)
    reference=GBFRSSelector().fit(X,y);ablation=CriterionAblationSelector().fit(X,y)
    assert np.array_equal(reference.selected_indices_,ablation.selected_indices_)
    assert np.array_equal(reference.ball_sizes_,ablation.ball_sizes_)
    assert np.allclose(reference.dependency_history_,ablation.dependency_history_)
    assert len(experiments())==11


def test_noise_is_confined_to_outer_training(synthetic_table,monkeypatch):
    import busbra.secondary as secondary
    from busbra.secondary import BenchmarkData,evaluate_secondary
    observed=[]
    def perturb(y,g,fold):
        observed.append((set(g),y.copy(),fold))
        return corrupt_training_labels(y,g,fold,.2)
    monkeypatch.setattr(secondary,'svm_grid',lambda:{'svm__C':[1.],'svm__kernel':['linear']})
    data=BenchmarkData(synthetic_table,tuple(f'original_{i}' for i in range(12)),'synthetic','synthetic','patient')
    result=evaluate_secondary(data,'baseline',n_jobs=1,training_label_transform=perturb)
    assert len(observed)==5
    for groups,y,fold in observed:
        assert groups==set(synthetic_table.loc[synthetic_table.fold5_kfold!=fold,'patient_id'])
        assert not groups & set(synthetic_table.loc[synthetic_table.fold5_kfold==fold,'patient_id'])
    assert np.array_equal(result.predictions.y_true,synthetic_table.diagnosis)
    assert all(f['training_label_perturbation']['flipped_groups']>0 for f in result.folds)
