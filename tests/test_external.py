import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


def load_tool(name):
    path=Path(__file__).resolve().parents[1]/'tools'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_external_diagnosis_comes_only_from_classification():
    extract=load_tool('extract_breast')
    table=pd.DataFrame({'Classification':['benign']*154+['malignant']*98+['normal']*4,
                        'BIRADS':[5]*256,'Interpretation':['misleading']*256})
    cohort,counts=extract.feature_labels(table)
    assert len(cohort)==252 and cohort.diagnosis.sum()==98 and counts['normal']==4
    table.loc[0,'Classification']='unknown'
    with pytest.raises(ValueError,match='diagnosis'):extract.feature_labels(table)


def test_external_loader_checks_source_hash_and_patient_uniqueness(tmp_path):
    external=load_tool('run_external')
    rng=np.random.RandomState(8)
    table=pd.DataFrame(rng.normal(size=(252,102)),columns=[f'original_synthetic_{i}' for i in range(102)])
    table['image_id']=[f'synthetic-image-{i}' for i in range(252)]
    table['patient_id']=[f'synthetic-person-{i}' for i in range(252)]
    table['diagnosis']=[0]*154+[1]*98
    path=tmp_path/'features.csv';table.to_csv(path,index=False)
    def provenance():
        (tmp_path/'provenance.json').write_text(json.dumps({'feature_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    provenance()
    data,_=external.load_external(tmp_path)
    assert len(data.feature_names)==102 and set(data.table.fold5_kfold)==set(range(1,6))
    assert np.array_equal(data.table.fold5_kfold,external.load_external(tmp_path)[0].table.fold5_kfold)
    table.loc[1,'patient_id']=table.loc[0,'patient_id'];table.to_csv(path,index=False)
    with pytest.raises(ValueError,match='fingerprint'):external.load_external(tmp_path)
    provenance()
    with pytest.raises(ValueError,match='coverage'):external.load_external(tmp_path)


def test_source_column_spelling_is_normalized_without_reading_predictor_metadata(monkeypatch):
    from types import SimpleNamespace
    extract=load_tool('extract_breast')
    source=pd.DataFrame({'CaseID':range(256),'Image filename':['synthetic']*256,
                         'Mask_tumor_filename':['synthetic']*256,'Classification':['benign']*256})
    monkeypatch.setattr(extract.pd,'ExcelFile',lambda path:SimpleNamespace(sheet_names=['source']))
    monkeypatch.setattr(extract.pd,'read_excel',lambda *args,**kwargs:source)
    normalized=extract.read_clinical('unused')
    assert {'Case_ID','Image_filename','Mask_tumor_filename','Classification'}<=set(normalized)


def test_download_retries_timeout_without_publishing_partial_data(tmp_path, monkeypatch):
    import io
    import urllib.error
    extract = load_tool('extract_breast')
    response = io.BytesIO(b'complete')
    response.headers = {'Content-Length': '8'}
    calls = []
    def open_source(*args, **kwargs):
        calls.append(kwargs['timeout'])
        if len(calls) == 1:
            raise urllib.error.URLError(TimeoutError())
        return response
    monkeypatch.setattr(extract.urllib.request, 'urlopen', open_source)
    monkeypatch.setattr(extract.time, 'sleep', lambda delay: None)
    target = tmp_path / 'release.zip'
    extract.download('https://example.test/release', target)
    assert target.read_bytes() == b'complete'
    assert calls == [60, 60]
    assert list(tmp_path.iterdir()) == [target]


def test_download_exhaustion_removes_partial_files(tmp_path, monkeypatch):
    import io
    extract = load_tool('extract_breast')
    def incomplete(*args, **kwargs):
        response = io.BytesIO(b'partial')
        response.headers = {'Content-Length': '100'}
        return response
    monkeypatch.setattr(extract.urllib.request, 'urlopen', incomplete)
    monkeypatch.setattr(extract.time, 'sleep', lambda delay: None)
    with pytest.raises(extract.http.client.IncompleteRead):
        extract.download('https://example.test/release', tmp_path / 'release.zip')
    assert not list(tmp_path.iterdir())


def test_download_does_not_retry_missing_release(tmp_path, monkeypatch):
    import urllib.error
    extract = load_tool('extract_breast')
    calls = []
    def missing(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('https://example.test', 404, 'missing', {}, None)
    monkeypatch.setattr(extract.urllib.request, 'urlopen', missing)
    with pytest.raises(urllib.error.HTTPError):
        extract.download('https://example.test/release', tmp_path / 'release.zip')
    assert len(calls) == 1
    assert not list(tmp_path.iterdir())


def test_download_preserves_existing_destination(tmp_path, monkeypatch):
    extract = load_tool('extract_breast')
    target = tmp_path / 'release.zip'
    target.write_bytes(b'prior source')
    def unexpected(*args, **kwargs):
        pytest.fail('Existing source must be checked before network access')
    monkeypatch.setattr(extract.urllib.request, 'urlopen', unexpected)
    with pytest.raises(FileExistsError):
        extract.download('https://example.test/release', target)
    assert target.read_bytes() == b'prior source'
