#!/usr/bin/env python3
"""Validate paired outer-fold predictions and build the BUS-BRA three-way scoreboard.

All methods MUST use the same frozen dataset and supplied patient-disjoint 5-fold split.
This script does not train models or alter input features.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'inputs' / 'BUSBRA_features_ready.csv'
OUT = ROOT / 'comparison_output'
METHODS = ['baseline', 'fuzzy_rough', 'gbfrs']
PROTOCOL = 'BUSBRA-V1-OUTER5-INNER3-MACROF1'
PRED_COLS = ['image_id','patient_id','outer_fold','y_true','y_pred']
SELECTED_COLS = ['outer_fold','feature_name']


def fail(message: str):
    raise ValueError(message)


def scores(df):
    true = df['y_true'].to_numpy().astype(int)
    pred = df['y_pred'].to_numpy().astype(int)
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    return {
        'n_images': len(df),
        'accuracy': accuracy_score(true, pred),
        'macro_f1': f1_score(true, pred, labels=[0, 1], average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(true, pred),
        'malignant_recall': tp / (tp + fn) if tp + fn else float('nan'),
        'benign_specificity': tn / (tn + fp) if tn + fp else float('nan'),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def main():
    global ROOT, DATA, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Private handoff root with inputs/ and submissions/")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    DATA = ROOT / "inputs" / "BUSBRA_features_ready.csv"
    OUT = ROOT / "comparison_output"
    if not DATA.is_file(): fail(f'Missing frozen dataset: {DATA}')
    raw = DATA.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    contract = json.loads((Path(__file__).resolve().parents[1] / 'evaluation/evaluation_contract.json').read_text())
    if digest != contract['dataset_sha256']: fail('Frozen dataset SHA-256 differs from the protocol.')
    source = pd.read_csv(DATA)
    feat_names = [c for c in source if c.startswith('original_') or c.startswith('wavelet-')]
    if len(source) != 1875 or len(feat_names) != 288: fail('Frozen dataset shape changed; stop before comparing.')
    if source.image_id.duplicated().any() or source[feat_names].isna().any().any(): fail('Invalid source dataset.')
    base = source[['image_id', 'patient_id', 'fold5_kfold','diagnosis']].rename(columns={'fold5_kfold':'outer_fold','diagnosis':'y_true'}).copy()
    if (base.groupby('patient_id').outer_fold.nunique() != 1).any(): fail('Source outer folds are not patient-disjoint.')
    base['outer_fold'] = base['outer_fold'].astype(int)
    base['y_true'] = base['y_true'].astype(int)
    base['image_id'] = base['image_id'].astype(str)
    base['patient_id'] = base['patient_id'].astype(str)
    OUT.mkdir(exist_ok=True)
    rows=[]; per_fold=[]; prediction_sets={}
    for method in METHODS:
        location = ROOT / 'submissions' / method
        pred_path = location / 'predictions.csv'
        features_path = location / 'selected_features.csv'
        info_path = location / 'run_info.json'
        for path in [pred_path, features_path, info_path]:
            if not path.exists(): fail(f'Missing required {method} file: {path}')
        meta = json.loads(info_path.read_text(encoding='utf-8'))
        if meta.get('protocol_id') != PROTOCOL: fail(f'{method}: protocol_id differs.')
        if meta.get('dataset_sha256') != digest: fail(f'{method}: dataset SHA-256 differs. Ensure SAME original CSV bytes.')
        if meta.get('method') != method: fail(f'{method}: wrong method name.')
        if meta.get('status') == 'TEMPLATE_ONLY_NOT_TRAINED': fail(f'{method}: template metadata was not replaced by a real experiment record.')
        p = pd.read_csv(pred_path)
        if any(c not in p for c in PRED_COLS): fail(f'{method}: prediction columns missing; required {PRED_COLS}')
        if len(p) != len(base): fail(f'{method}: expected 1875 predictions, got {len(p)}.')
        if p.image_id.isna().any() or p.image_id.duplicated().any(): fail(f'{method}: missing/duplicate image_id.')
        if not set(p.y_true.dropna().unique()).issubset({0,1}) or not set(p.y_pred.dropna().unique()).issubset({0,1}) or p[['y_true','y_pred']].isna().any().any():
            fail(f'{method}: y values must be 0 or 1 with no nulls.')
        p = p[PRED_COLS].copy()
        p['image_id'] = p.image_id.astype(str)
        p['patient_id'] = p.patient_id.astype(str)
        joined = base.merge(p, on='image_id', how='outer', indicator=True, suffixes=('_expected','_submitted'),validate='one_to_one')
        if not (joined._merge == 'both').all(): fail(f'{method}: image IDs do not exactly match reference.')
        for col in ['patient_id','outer_fold','y_true']:
            if not (joined[col+'_expected'].astype(str) == joined[col+'_submitted'].astype(str)).all():
                fail(f'{method}: {col} mismatch from reference. Possible leakage/wrong labels.')
        p = base.merge(p[['image_id','y_pred']],on='image_id',validate='one_to_one')
        prediction_sets[method] = p
        selected = pd.read_csv(features_path)
        if any(c not in selected for c in SELECTED_COLS): fail(f'{method}: selected feature file requires {SELECTED_COLS}')
        if selected[SELECTED_COLS].isna().any().any(): fail(f'{method}: null selected-feature entries.')
        if selected[SELECTED_COLS].duplicated().any(): fail(f'{method}: duplicate selected-feature entry.')
        if not set(selected.feature_name).issubset(feat_names): fail(f'{method}: a selected feature is not in the original 288 columns.')
        if set(selected.outer_fold.astype(int)) != {1,2,3,4,5}: fail(f'{method}: one or more outer folds missing from selected features.')
        counts = selected.groupby('outer_fold').feature_name.nunique()
        if method == 'baseline' and any(counts != len(feat_names)):
            fail('Baseline is required to use ALL 288 features in each fold.')
        if (counts < 1).any(): fail(f'{method}: zero selected features in a fold.')
        full = scores(p)
        fold_metrics=[]
        for f in range(1,6):
            part = p.loc[p.outer_fold==f]
            v = scores(part)
            v.update({'method':method,'outer_fold':f,'n_selected_features':int(counts[f])})
            fold_metrics.append(v)
            per_fold.append(v)
        r={'method':method, 'n_predictions':len(p),'mean_selected_features':round(float(counts.mean()),2),
           'mean_feature_reduction_pct':round(100*(1-float(counts.mean())/len(feat_names)),2)}
        for metric in ['accuracy','macro_f1','balanced_accuracy','malignant_recall','benign_specificity']:
            arr=[v[metric] for v in fold_metrics]
            r[metric+'_mean']=np.mean(arr)
            r[metric+'_sd']=np.std(arr,ddof=1)
            r[metric+'_pooled']=full[metric]
        rows.append(r)
    dfm = pd.DataFrame(rows)
    fdf = pd.DataFrame(per_fold)
    dfm.to_csv(OUT/'comparison_summary.csv',index=False)
    fdf.to_csv(OUT/'per_fold_metrics.csv',index=False)
    with (OUT/'comparison_report.md').open('w',encoding='utf-8') as fw:
        fw.write('# BUS-BRA locked-protocol comparison\n\n')
        fw.write(f'Dataset SHA-256: `{digest}`. Outer folds: supplied patient-disjoint `fold5_kfold` (1-5).\n\n')
        fw.write('Primary selection/reporting metric: macro-F1. Accuracy is secondary. Five-fold mean and sample SD are shown; pooled scores are also provided.\n\n')
        fw.write('| Method | Accuracy (mean +/- SD) | Macro-F1 (mean +/- SD) | Malignant recall (mean) | Benign specificity (mean) | Mean features |\n')
        fw.write('|---|---:|---:|---:|---:|---:|\n')
        for row in rows:
            fw.write(f"| {row['method']} | {row['accuracy_mean']:.3f} +/- {row['accuracy_sd']:.3f} | {row['macro_f1_mean']:.3f} +/- {row['macro_f1_sd']:.3f} | {row['malignant_recall_mean']:.3f} | {row['benign_specificity_mean']:.3f} | {row['mean_selected_features']:.1f} |\n")
        fw.write('\n## Interpretation safeguards\n\n')
        fw.write('- The checker validates prediction IDs, groups, fold assignments, labels, allowed selected names, dataset hash and method metadata.\n')
        fw.write('- It **cannot** independently prove that preprocessing, selector fitting and hyperparameter tuning were nested correctly. Code review is required.\n')
        fw.write('- Never adjust any model after examining outer-fold predictions without declaring a new experiment. No accuracy level is guaranteed.\n')
    print('Validated 3 submissions. Results written to',OUT)
    print(dfm[['method','accuracy_mean','macro_f1_mean','malignant_recall_mean','mean_selected_features']].to_string(index=False))

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'COMPARISON BLOCKED: {exc}',file=sys.stderr)
        sys.exit(2)
