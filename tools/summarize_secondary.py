#!/usr/bin/env python3
"""Validate private secondary exports and publish aggregate measurements only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from busbra.data import load_dataset
from busbra.evaluation import EvaluationResult, classification_metrics
from busbra.secondary import BenchmarkData, PROTOCOL, aggregate_secondary, load_wdbc


def validate_run(root, data, expected_protocol=PROTOCOL):
    record = json.loads((root/'experiment.json').read_text())
    if record['status'] not in ('COMPLETED', 'COMPLETED_WITH_REPRODUCTION_BLOCKERS'):
        raise ValueError('Refuse incomplete secondary experiments')
    assert record['dataset_sha256'] == data.sha256
    assert record['protocol_id'] == expected_protocol
    summaries, predictions, sets = {}, {}, {}
    for method, reported in record['methods'].items():
        folder = root/'submissions'/method
        details = json.loads((folder/'internal_run.json').read_text())
        assert details['dataset_sha256'] == data.sha256 and details['method'] == method
        assert details['protocol_id'] == expected_protocol and details['status'] == 'COMPLETED'
        for name, digest in details['source_sha256'].items():
            assert hashlib.sha256((folder/'code'/'busbra'/name).read_bytes()).hexdigest() == digest
        prediction = pd.read_csv(folder/'predictions.csv')
        selected = pd.read_csv(folder/'selected_features.csv')
        expected = data.table
        assert len(prediction) == len(expected) and prediction.image_id.is_unique
        aligned = prediction.set_index('image_id').loc[expected.image_id]
        for col, target in [('patient_id','patient_id'),('outer_fold','fold5_kfold'),('y_true','diagnosis')]:
            assert np.array_equal(aligned[col].astype(str).to_numpy(), expected[target].astype(str).to_numpy())
        assert set(aligned.y_pred) <= {0,1}
        assert not selected.duplicated().any() and set(selected.outer_fold) == set(range(1,6))
        assert set(selected.feature_name) <= set(data.feature_names)
        assert [f['outer_fold'] for f in details['folds']] == list(range(1,6))
        for fold in details['folds']:
            f = fold['outer_fold']
            subset = aligned.loc[aligned.outer_fold == f]
            metrics = classification_metrics(subset.y_true, subset.y_pred)
            assert all(np.isclose(metrics[k], fold['metrics'][k], rtol=0, atol=1e-12) for k in metrics)
            assert fold['selected_feature_count'] == int((selected.outer_fold == f).sum())
            assert fold['n_candidates'] == 32*(len(details['selector_grid'])-len(fold['rejected_selector_settings']))
            assert len(fold['search_results']) == fold['n_candidates']
            assert fold['inner_seed'] == 42+f
            assert fold['test_rows'] == len(subset) and fold['training_rows'] == len(expected)-len(subset)
        result = EvaluationResult(prediction, selected, details['folds'], details)
        actual = aggregate_secondary(result, data)
        assert actual == reported, 'Manifest aggregate differs from private outputs'
        summaries[method] = actual
        predictions[method] = aligned.y_pred.to_numpy()
        sets[method] = [set(selected.loc[selected.outer_fold == f,'feature_name']) for f in range(1,6)]
    agreement = None
    if {'gbfrs','author_gbfrs'} <= sets.keys():
        agreement = {'prediction_agreement_fraction':float(np.mean(predictions['gbfrs'] == predictions['author_gbfrs'])),
                     'mean_feature_jaccard':float(np.mean([len(a&b)/len(a|b) for a,b in zip(sets['gbfrs'],sets['author_gbfrs'])]))}
    return {'dataset_sha256':data.sha256, 'unit':data.unit, 'n_rows':len(data.table),
            'n_features':len(data.feature_names), 'methods':summaries,
            'reference_agreement':agreement, 'failures':record['failures'], 'validation':'passed: alignment, labels, folds, metrics, counts and snapshot hashes'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--busbra-data',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    original = load_dataset(args.busbra_data)
    bus = BenchmarkData(original.table,original.feature_names,original.sha256,'busbra','patient')
    payload = {'protocol_id':PROTOCOL, 'scope':'Secondary descriptive experiments; not a public preregistration or external ultrasound validation.',
               'wdbc_source':'UCI WDBC DOI 10.24432/C5DW2B, bundled with scikit-learn 1.6.1; cytology, case-level folds',
               'search_budget':'96 joint candidates for ANOVA/MI/L1; 32 for fixed selectors; not matched compute',
               'datasets':{'busbra':validate_run(args.root/'busbra',bus), 'wdbc':validate_run(args.root/'wdbc',load_wdbc())}}
    if (args.root/'busbra-author'/'experiment.json').exists():
        author = validate_run(args.root/'busbra-author',bus)
        payload['datasets']['busbra']['methods'].update(author['methods'])
        payload['datasets']['busbra']['failures'].update(author['failures'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print('Validated aggregate-only secondary results:',args.output)


if __name__ == '__main__': main()
