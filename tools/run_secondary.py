#!/usr/bin/env python3
"""Run separately specified additional baselines or the independent WDBC benchmark."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import signal

from busbra.data import load_dataset
from busbra.export import export_submission
from busbra.secondary import (BenchmarkData, PROTOCOL, NEW_BASELINES, WDBC_METHODS,
                              load_wdbc,evaluate_secondary,aggregate_secondary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['busbra','wdbc'],required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--author-code',type=Path)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--methods',nargs='+')
    p.add_argument('--method-timeout-seconds',type=int,default=900,
                   help='Wall-clock limit per method; timed-out methods are explicitly recorded')
    args=p.parse_args()
    if args.output.exists():
        p.error('Output already exists; preserve prior experiments')
    if args.method_timeout_seconds < 1:p.error('Method timeout must be positive')
    if args.dataset=='busbra':
        if args.data is None:p.error('--data is required for BUS-BRA')
        source=load_dataset(args.data)
        data=BenchmarkData(source.table,source.feature_names,source.sha256,'busbra','patient')
        methods=args.methods or list(NEW_BASELINES)
    else:
        data=load_wdbc()
        methods=args.methods or [m for m in WDBC_METHODS if m!='author_gbfrs' or args.author_code is not None]
    if any(m not in WDBC_METHODS for m in methods):p.error('Unknown secondary method')
    args.output.mkdir(parents=True)
    record={'protocol_id':PROTOCOL,'dataset':data.name,'dataset_sha256':data.sha256,
            'unit':data.unit,'started_utc':datetime.now(timezone.utc).isoformat(),
            'method_timeout_seconds':args.method_timeout_seconds,
            'status':'RUNNING','methods':{},'failures':{}}
    manifest=args.output/'experiment.json'
    def progress(dataset,method,fold,features,seconds):
        print(f'{dataset}/{method}: fold {fold}/5, {features} features, {seconds:.1f}s',flush=True)
    try:
        for method in methods:
            manifest.write_text(json.dumps(record,indent=2)+'\n')
            print(f'Starting {data.name}/{method}',flush=True)
            previous_handler = signal.getsignal(signal.SIGALRM)
            def timeout(signum, frame):
                raise TimeoutError(f'Method exceeded {args.method_timeout_seconds}s wall-clock limit')
            try:
                signal.signal(signal.SIGALRM, timeout)
                signal.alarm(args.method_timeout_seconds)
                result=evaluate_secondary(data,method,n_jobs=args.jobs,author_path=args.author_code,progress=progress)
                signal.alarm(0)
                export_submission(result,args.output/'submissions'/method)
                record['methods'][method]=aggregate_secondary(result,data)
            except Exception as error:
                record['failures'][method]={'type':type(error).__name__,'message':str(error)}
                print(f'{data.name}/{method}: FAILED explicitly: {type(error).__name__}: {error}',flush=True)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)
        record['status']='COMPLETED' if not record['failures'] else 'COMPLETED_WITH_REPRODUCTION_BLOCKERS'
        record['completed_utc']=datetime.now(timezone.utc).isoformat()
    finally:
        manifest.write_text(json.dumps(record,indent=2)+'\n')
    print('Secondary benchmark finished:',manifest,flush=True)
    return 0 if not record['failures'] else 2


if __name__=='__main__':raise SystemExit(main())
