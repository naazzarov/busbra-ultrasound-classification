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
