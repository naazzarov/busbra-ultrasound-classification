import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from busbra.secondary import (load_wdbc, RankedSelector, L1Selector, selector_configs,
                              make_secondary_selector, _load_author_module, AUTHOR_SHA256)


def test_wdbc_encoding_and_independent_case_folds():
    data=load_wdbc()
    assert data.table.shape[0]==569 and len(data.feature_names)==30
    assert data.table.diagnosis.sum()==212
    assert data.unit=='case'
    assert set(data.table.fold5_kfold)=={1,2,3,4,5}
    assert data.table.patient_id.is_unique
    assert data.sha256==load_wdbc().sha256


def test_baseline_grid_is_fixed_and_rank_selector_is_training_only():
    assert selector_configs('anova',288)==[{'k':5},{'k':10},{'k':20}]
    assert selector_configs('anova',4)==[{'k':4}]
    rng=np.random.RandomState(3); X=rng.normal(size=(60,8)); y=(X[:,0]>0).astype(int)
    selector=RankedSelector('anova',1).fit(X,y)
    assert selector.get_support(indices=True).tolist()==[0]
    assert selector.transform(X[:4]).shape==(4,1)
    mi=RankedSelector('mutual_information',2).fit(X,y)
    assert 0 in mi.get_support(indices=True)


def test_author_code_is_optional_and_hash_pinned(tmp_path):
    with pytest.raises(ValueError,match='required'):
        make_secondary_selector('author_gbfrs',{},None)
    fake=tmp_path/'fake.py';fake.write_text('raise AssertionError("must not execute")')
    with pytest.raises(ValueError,match='pinned'):
        _load_author_module(str(fake))


def test_secondary_nested_selection_never_fits_held_out_groups(synthetic_table,monkeypatch,tmp_path):
    import busbra.secondary as secondary
    from busbra.secondary import BenchmarkData,evaluate_secondary
    observed=[]
    class MarkerScaler:
        def fit(self,X):return self
        def transform(self,X):return X
    class SpyRank(RankedSelector):
        def fit(self,X,y):
            observed.append(set(X[:,0].astype(int)))
            return super().fit(X,y)
    monkeypatch.setattr(secondary,'StandardScaler',MarkerScaler)
    monkeypatch.setattr(secondary,'make_secondary_selector',lambda method,config,path=None:SpyRank(method,**config))
    names=tuple(f'original_{i}' for i in range(12))
    data=BenchmarkData(synthetic_table,names,'synthetic','synthetic','patient')
    result=evaluate_secondary(data,'anova',n_jobs=1)
    assert len(observed)==50
    for fold in range(1,6):
        expected=set(synthetic_table.loc[synthetic_table.fold5_kfold!=fold,'patient_id'])
        fits=observed[(fold-1)*10:fold*10]
        assert all(groups<expected for groups in fits[:-1])
        assert fits[-1]==expected
    assert len(result.predictions)==120
    assert [f['n_candidates'] for f in result.folds]==[96]*5
    # Validate the exact export, then reject a plausible but altered reported score.
    import importlib.util
    import json
    from pathlib import Path
    from busbra.export import export_submission
    from busbra.secondary import aggregate_secondary, PROTOCOL
    folder=tmp_path/'submissions/anova'
    export_submission(result,folder)
    assert (folder/'code/run_secondary.py').is_file()
    assert 'tools/run_secondary.py' in (folder/'README.md').read_text()
    record={'status':'COMPLETED','dataset_sha256':data.sha256,'protocol_id':PROTOCOL,
            'methods':{'anova':aggregate_secondary(result,data)},'failures':{}}
    (tmp_path/'experiment.json').write_text(json.dumps(record))
    spec=importlib.util.spec_from_file_location('secondary_summary',Path(__file__).resolve().parents[1]/'tools/summarize_secondary.py')
    summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
    assert 'anova' in summary.validate_run(tmp_path,data)['methods']
    altered=json.loads((folder/'internal_run.json').read_text())
    altered['folds'][0]['metrics']['f1_macro']=.12345
    (folder/'internal_run.json').write_text(json.dumps(altered))
    with pytest.raises(AssertionError): summary.validate_run(tmp_path,data)


def test_secondary_runner_reports_reference_timeout(tmp_path,monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    import sys
    spec=importlib.util.spec_from_file_location('secondary_runner',Path(__file__).resolve().parents[1]/'tools/run_secondary.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    def fail(*args,**kwargs):raise TimeoutError('Synthetic reference deadline')
    monkeypatch.setattr(runner,'evaluate_secondary',fail)
    monkeypatch.setattr(sys,'argv',['run_secondary','--dataset','wdbc','--methods','author_gbfrs',
                                  '--output',str(tmp_path/'run')])
    assert runner.main()==2
    result=json.loads((tmp_path/'run/experiment.json').read_text())
    assert result['status']=='COMPLETED_WITH_REPRODUCTION_BLOCKERS'
    assert result['failures']['author_gbfrs']['type']=='TimeoutError'
    assert result['methods']=={}
