#!/usr/bin/env python3
"""Evaluate retrained pipelines on independently extracted BrEaST data; aggregate only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from busbra.secondary import BenchmarkData,evaluate_secondary,aggregate_secondary
from busbra.evaluation import classification_metrics

PROTOCOL='BREAST-INDEPENDENT-20261028-V1'
METHODS=('baseline','fuzzy_rough','gbfrs','anova','mutual_information','l1_logistic')


def load_external(root):
    source=json.loads((root/'provenance.json').read_text())
    path=root/'features.csv';sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=source['feature_csv_sha256']:raise ValueError('Extracted table fingerprint mismatch')
    table=pd.read_csv(path);names=tuple(c for c in table if c.startswith('original_'))
    if len(table)!=252 or len(names)!=102 or not table.patient_id.is_unique or not table.image_id.is_unique:
        raise ValueError('External patient/feature coverage mismatch')
    if table.diagnosis.value_counts().to_dict()!={0:154,1:98}:raise ValueError('External diagnosis count mismatch')
    if not np.isfinite(table[list(names)].to_numpy(dtype=float)).all():raise ValueError('Nonfinite predictors')
    for fold,(_,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(table,table.diagnosis),1):
        table.loc[test,'fold5_kfold']=fold
    table['fold5_kfold']=table.fold5_kfold.astype(int)
    return BenchmarkData(table,names,sha,'breast','patient'),source


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',type=int,default=2)
    args=p.parse_args()
    data,provenance=load_external(args.root)
    summary={'protocol_id':PROTOCOL,'status':'RUNNING','scope':'Independent ultrasound cohort method evaluation, separately retrained; not frozen-model external validation',
             'provenance':provenance,'methods':{},'failures':{}}
    def progress(dataset,method,fold,features,seconds):print(f'{method}: fold {fold}/5, {features} features',flush=True)
    for method in METHODS:
        try:
            result=evaluate_secondary(data,method,n_jobs=args.jobs,progress=progress,protocol_id=PROTOCOL)
            aligned=result.predictions.set_index('image_id').loc[data.table.image_id]
            assert np.array_equal(aligned.y_true,data.table.diagnosis)
            assert np.array_equal(aligned.outer_fold,data.table.fold5_kfold)
            assert np.array_equal(aligned.patient_id,data.table.patient_id)
            assert result.predictions.image_id.is_unique and len(result.predictions)==252
            for f in result.folds:
                selected=aligned.loc[aligned.outer_fold==f['outer_fold']]
                actual=classification_metrics(selected.y_true,selected.y_pred)
                assert all(np.isclose(actual[k],f['metrics'][k],rtol=0,atol=1e-12) for k in actual)
            summary['methods'][method]=aggregate_secondary(result,data)
            # Full audits are temporary private runner data; never upload these rows.
            private=args.root/'private_runs'/method;private.mkdir(parents=True)
            result.predictions.to_csv(private/'predictions.csv',index=False)
            (private/'internal_run.json').write_text(json.dumps({**result.metadata,'folds':result.folds},indent=2)+'\n')
        except Exception as error:
            summary['failures'][method]={'type':type(error).__name__,'reason':'Method did not complete; no substitute score reported'}
            print('Method failed:',method,type(error).__name__,flush=True)
    summary['status']='COMPLETED' if not summary['failures'] else 'COMPLETED_WITH_REPRODUCTION_BLOCKERS'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('PUBLIC_AGGREGATE_JSON_BEGIN',flush=True)
    print(json.dumps(summary,allow_nan=False),flush=True)
    print('PUBLIC_AGGREGATE_JSON_END',flush=True)
    return 0 if not summary['failures'] else 2


if __name__=='__main__':raise SystemExit(main())
