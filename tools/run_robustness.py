#!/usr/bin/env python3
"""Execute prespecified robustness experiments without changing primary results."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import signal
from busbra.data import load_dataset
from busbra.export import export_submission
from busbra.secondary import BenchmarkData,evaluate_secondary,aggregate_secondary
from busbra.robustness import PROTOCOL,experiments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experiments',nargs='+');p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--method-timeout-seconds',type=int,default=3600)
    args=p.parse_args()
    if args.output.exists():p.error('Output already exists; preserve experiment history')
    if args.method_timeout_seconds<1:p.error('Timeout must be positive')
    planned=experiments();mapping={x.name:x for x in planned}
    if args.experiments and not set(args.experiments)<=mapping.keys():p.error('Unknown prespecified experiment')
    selected=[mapping[name] for name in args.experiments] if args.experiments else planned
    original=load_dataset(args.data)
    data=BenchmarkData(original.table,original.feature_names,original.sha256,'busbra','patient')
    args.output.mkdir(parents=True)
    record={'protocol_id':PROTOCOL,'dataset':'busbra','unit':'patient','dataset_sha256':data.sha256,
            'started_utc':datetime.now(timezone.utc).isoformat(),'status':'RUNNING',
            'plan':[asdict(x) for x in selected],'methods':{},'failures':{}}
    manifest=args.output/'experiment.json'
    def save():manifest.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    def progress(dataset,method,fold,features,seconds):print(f'{current.name}: fold {fold}/5, {features} predictors, {seconds:.1f}s',flush=True)
    def deadline(signum,frame):raise TimeoutError('Declared method wall-clock limit exceeded')
    try:
        for current in selected:
            record['current_experiment']=current.name;save()
            print('Starting',current.name,flush=True)
            old=signal.getsignal(signal.SIGALRM)
            try:
                signal.signal(signal.SIGALRM,deadline);signal.alarm(args.method_timeout_seconds)
                result=evaluate_secondary(data,current.method,n_jobs=args.jobs,progress=progress,
                    selector_factory=current.selector,training_label_transform=current.labels if current.rate else None,
                    protocol_id=PROTOCOL)
                signal.alarm(0)
                result.metadata.update(method=current.name,base_method=current.method,experiment_spec=asdict(current))
                export_submission(result,args.output/'submissions'/current.name)
                record['methods'][current.name]=aggregate_secondary(result,data)
            except Exception as error:
                record['failures'][current.name]={'type':type(error).__name__,'message':str(error)}
                print('FAILED',current.name,type(error).__name__,str(error),flush=True)
            finally:
                signal.alarm(0);signal.signal(signal.SIGALRM,old)
            save()
        record.pop('current_experiment',None)
        record['status']='COMPLETED' if not record['failures'] else 'COMPLETED_WITH_REPRODUCTION_BLOCKERS'
        record['completed_utc']=datetime.now(timezone.utc).isoformat()
    finally:save()
    print('Robustness study finished:',manifest,flush=True)
    return 0 if not record['failures'] else 2


if __name__=='__main__':raise SystemExit(main())
