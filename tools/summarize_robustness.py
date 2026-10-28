#!/usr/bin/env python3
"""Validate all private robustness runs and publish aggregate sensitivity results."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from busbra.data import load_dataset
from busbra.secondary import BenchmarkData
from busbra.robustness import PROTOCOL,experiments,corrupt_training_labels
from summarize_secondary import validate_run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();original=load_dataset(args.data)
    data=BenchmarkData(original.table,original.feature_names,original.sha256,'busbra','patient')
    methods={};failures={}
    plan={e.name:e for e in experiments()}
    for name in ('noise','sensitivity'):
        run=args.root/name
        validated=validate_run(run,data,expected_protocol=PROTOCOL)
        for method in validated['methods']:
            spec=plan[method]
            details=json.loads((run/'submissions'/method/'internal_run.json').read_text())
            assert details['experiment_spec']==asdict(spec) and details['base_method']==spec.method
            if spec.rate:
                for fold in details['folds']:
                    train=data.table.loc[data.table.fold5_kfold!=fold['outer_fold']]
                    _,audit=corrupt_training_labels(train.diagnosis.to_numpy(),train.patient_id.to_numpy(),fold['outer_fold'],spec.rate)
                    assert fold['training_label_perturbation']==audit
                validated['methods'][method]['mean_actual_flipped_group_fraction']=float(np.mean([
                    f['training_label_perturbation']['flipped_groups']/f['training_label_perturbation']['training_groups'] for f in details['folds']]))
            validated['methods'][method]['specification']=asdict(spec)
        assert not set(methods)&set(validated['methods'])
        methods.update(validated['methods']);failures.update(validated['failures'])
    assert set(methods)|set(failures)==set(plan)
    summary={'protocol_id':PROTOCOL,'status':'COMPLETED' if not failures else 'COMPLETED_WITH_REPRODUCTION_BLOCKERS',
             'scope':'Exploratory robustness and controlled ablation; primary definitions unchanged; no outer-based selection or confirmatory inference',
             'dataset_sha256':data.sha256,'completed_experiments':len(methods),'planned_experiments':len(plan),
             'methods':methods,'failures':failures,
             'validation':'passed: source snapshots, OOF alignment, fold metrics, selected features, candidate counts and deterministic training-label perturbations'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('Validated robustness experiments:',len(methods),'completed;',len(failures),'explicit blockers')


if __name__=='__main__':main()
