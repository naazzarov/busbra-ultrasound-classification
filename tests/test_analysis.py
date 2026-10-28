import numpy as np
import pytest
from sklearn.metrics import f1_score

from busbra.analysis import metrics_from_counts, selection_stability, paired_patient_bootstrap


def test_macro_f1_from_counts_matches_sklearn():
    y = np.array([0,0,0,1,1])
    pred = np.array([0,0,1,0,1])
    assert metrics_from_counts([2,1,1,1]) == pytest.approx(f1_score(y,pred,average='macro'))


def test_stability_has_identical_and_disjoint_limits():
    assert selection_stability([{'a'},{'a'}], ['a','b'])['nogueira'] == 1
    assert selection_stability([{'a'},{'b'}], ['a','b'])['mean_pairwise_jaccard'] == 0
    assert selection_stability([{'a','b'},{'a','b'}], ['a','b'])['nogueira'] == 1
    with pytest.raises(ValueError):
        selection_stability([{'a'},{'unknown'}], ['a','b'])


def test_paired_patient_bootstrap_preserves_equal_predictions_and_seed():
    y = np.repeat([0,0,1,1],2)
    groups = np.repeat(['a','b','c','d'],2)
    pred = np.array([0,0,0,1,1,1,1,0])
    args = (y, {'a':pred,'b':pred}, groups)
    first = paired_patient_bootstrap(*args, n_bootstrap=100)
    assert first == paired_patient_bootstrap(*args, n_bootstrap=100)
    assert first[1]['b_minus_a'] == [0,0]
    assert all(0 <= value <= 1 for value in first[0]['a'])


def test_bootstrap_rejects_inconsistent_patient_labels():
    with pytest.raises(ValueError,match='consistent diagnosis'):
        paired_patient_bootstrap([0,1], {'a':[0,1]}, ['same','same'], n_bootstrap=100)


def test_summary_rejects_swapped_fold_labels_and_exports_no_subject_keys(tmp_path, synthetic_table):
    import json
    import pandas as pd
    from busbra.data import ValidatedDataset
    from busbra.analysis import summarize, METHODS
    from busbra.protocol import read_contract
    names=tuple(c for c in synthetic_table if c.startswith('original_'))
    data=ValidatedDataset(synthetic_table,names,'synthetic')
    for method in METHODS:
        path=tmp_path/method; path.mkdir()
        pd.DataFrame({'image_id':synthetic_table.image_id,'patient_id':synthetic_table.patient_id,
                      'outer_fold':synthetic_table.fold5_kfold,'y_true':synthetic_table.diagnosis,
                      'y_pred':synthetic_table.diagnosis}).to_csv(path/'predictions.csv',index=False)
        selected=names if method=='baseline' else names[:3]
        pd.DataFrame([{'outer_fold':f,'feature_name':n} for f in range(1,6) for n in selected]).to_csv(path/'selected_features.csv',index=False)
        pd.DataFrame({'outer_fold':range(1,6),'total_seconds':[1.]*5}).to_csv(path/'timings.csv',index=False)
        (path/'run_info.json').write_text(json.dumps({'method':method,'dataset_sha256':'synthetic',
            'protocol_id':read_contract()['protocol_id'],'status':'COMPLETED','versions':{},'platform':'synthetic'}))
    result=summarize(data,tmp_path,n_bootstrap=100)
    assert result['methods']['baseline']['pooled']['f1_macro']==1
    assert result['paired_effects']['gbfrs_minus_baseline']['patient_bootstrap_95ci']==[0,0]
    serialized=json.dumps(result)
    assert 'image_id' not in serialized and 'patient_id' not in serialized
    broken=pd.read_csv(tmp_path/'gbfrs/predictions.csv')
    broken.loc[0,'outer_fold']=2
    broken.to_csv(tmp_path/'gbfrs/predictions.csv',index=False)
    with pytest.raises(ValueError,match='alignment'):
        summarize(data,tmp_path,n_bootstrap=100)
