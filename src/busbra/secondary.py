"""Separately declared secondary benchmarks; never relax the primary CSV gate."""
from dataclasses import dataclass
import contextlib
from functools import lru_cache
import hashlib
from importlib.metadata import version
import importlib.util
import io
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .analysis import selection_stability
from .evaluation import EvaluationResult, _score_candidate, choose_candidate, classification_metrics
from .protocol import svm_grid
from .selectors import make_selector

PROTOCOL = 'BUSBRA-SECONDARY-20261028-V1'
AUTHOR_SHA256 = '8f675c052aaaab5760a853112990022bf4f5ba7afa7bb3b54d903e9467ec5c9b'
RANK_K = (5, 10, 20)
L1_C = (.01, .1, 1.)
NEW_BASELINES = ('anova', 'mutual_information', 'l1_logistic')
WDBC_METHODS = ('baseline', 'fuzzy_rough', 'gbfrs', *NEW_BASELINES, 'author_gbfrs')


@dataclass(frozen=True)
class BenchmarkData:
    table: pd.DataFrame
    feature_names: tuple[str, ...]
    sha256: str
    name: str
    unit: str


def load_wdbc():
    data = load_breast_cancer()
    if list(data.target_names) != ['malignant', 'benign'] or data.data.shape != (569, 30):
        raise ValueError('Unexpected bundled WDBC version or class encoding')
    X = np.asarray(data.data, dtype='<f8')
    y = (data.target == 0).astype(np.int64)  # malignant=1; sklearn uses malignant=0.
    names = tuple(str(name) for name in data.feature_names)
    digest = hashlib.sha256(X.tobytes() + y.astype('<i8').tobytes() + '\n'.join(names).encode()).hexdigest()
    table = pd.DataFrame(X, columns=names)
    table['diagnosis'] = y
    table['image_id'] = [f'wdbc-row-{i}' for i in range(len(y))]
    table['patient_id'] = table.image_id  # case identifier, not a verified patient mapping.
    table['fold5_kfold'] = 0
    for f, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1):
        table.loc[test, 'fold5_kfold'] = f
    return BenchmarkData(table, names, digest, 'wdbc', 'case')


class RankedSelector(TransformerMixin, BaseEstimator):
    def __init__(self, kind='anova', k=10):
        self.kind, self.k = kind, k

    def fit(self, X, y):
        if self.kind == 'anova':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                scores = f_classif(X, y)[0]
        elif self.kind == 'mutual_information':
            scores = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
        else:
            raise ValueError('Unknown rank selector')
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        ranked = np.argsort(-scores, kind='stable')
        self.selected_indices_ = np.sort(ranked[:min(self.k, X.shape[1])])
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def get_support(self, indices=False):
        if not indices:
            raise ValueError('This secondary selector returns indices only')
        return self.selected_indices_.copy()


class L1Selector(TransformerMixin, BaseEstimator):
    def __init__(self, C=.1):
        self.C = C

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model = LogisticRegression(C=self.C, penalty='l1', solver='liblinear', max_iter=10000, random_state=42).fit(X, y)
        self.selected_indices_ = np.flatnonzero(np.abs(model.coef_[0]) > 1e-5)
        if not len(self.selected_indices_):
            raise ZeroFeatures('L1 setting selects no predictors on this training partition')
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def get_support(self, indices=False):
        if not indices:
            raise ValueError('This secondary selector returns indices only')
        return self.selected_indices_.copy()


class ZeroFeatures(ValueError):
    pass


@lru_cache(maxsize=2)
def _load_author_module(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != AUTHOR_SHA256:
        raise ValueError('Author source differs from the pinned inspected script')
    spec = importlib.util.spec_from_file_location('_private_author_gbfrs', path)
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        spec.loader.exec_module(module)
    return module


class AuthorSelector(TransformerMixin, BaseEstimator):
    """Call pinned private author code unchanged; no source is bundled here."""
    def __init__(self, path):
        self.path = path

    def fit(self, X, y):
        module = _load_author_module(str(Path(self.path).resolve()))
        span = np.ptp(X, axis=0)
        span[span == 0] = 1
        geometry = (X-X.min(axis=0))/span
        augmented = np.column_stack([geometry, y, np.arange(len(y))])
        state = np.random.get_state()
        try:
            np.random.seed(42)
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                selected = module.attribute_reduce(augmented, pur=.9)
        finally:
            np.random.set_state(state)
        indices = np.asarray(selected, dtype=int)
        if (not len(indices) or len(set(indices)) != len(indices)
                or (indices < 0).any() or (indices >= X.shape[1]).any()):
            raise ValueError('Author selector returned invalid feature indices')
        self.selected_indices_ = indices
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def get_support(self, indices=False):
        if not indices:
            raise ValueError('This secondary selector returns indices only')
        return self.selected_indices_.copy()


def selector_configs(method, p):
    if method in ('anova', 'mutual_information'):
        return [{'k': k} for k in dict.fromkeys(min(k, p) for k in RANK_K)]
    if method == 'l1_logistic':
        return [{'C': C} for C in L1_C]
    return [{}]


def make_secondary_selector(method, config, author_path=None):
    if method in ('anova', 'mutual_information'):
        return RankedSelector(method, **config)
    if method == 'l1_logistic':
        return L1Selector(**config)
    if method == 'author_gbfrs':
        if author_path is None:
            raise ValueError('Unchanged private author code is required for this optional method')
        return AuthorSelector(str(author_path))
    return make_selector(method)


def _prepare_secondary(X, y, method, config, author_path, selector_factory=None):
    scaler = StandardScaler().fit(X)
    transformed = scaler.transform(X)
    factory = selector_factory or make_secondary_selector
    selector = factory(method, config, author_path)
    start = perf_counter()
    if selector is not None:
        transformed = selector.fit_transform(transformed, y)
        indices = selector.get_support(indices=True)
    else:
        indices = np.arange(X.shape[1])
    return scaler, selector, transformed, indices, perf_counter()-start if selector is not None else 0.


def _transform(scaler, selector, X):
    value = scaler.transform(X)
    return selector.transform(value) if selector is not None else value


def evaluate_secondary(data, method, *, n_jobs=2, author_path=None, progress=None,
                       selector_factory=None, training_label_transform=None,
                       protocol_id=PROTOCOL):
    if n_jobs < 1:
        raise ValueError('n_jobs must be positive')
    table = data.table
    names = list(data.feature_names)
    X = table[names].to_numpy(dtype=float)
    y = table.diagnosis.to_numpy(dtype=int)
    groups = table.patient_id.to_numpy()
    configs = selector_configs(method, X.shape[1])
    grid = list(ParameterGrid(svm_grid()))
    predictions = np.full(len(y), -1, dtype=int)
    held_out_count = np.zeros(len(y), dtype=int)
    selected_rows, folds = [], []
    with threadpool_limits(limits=1):
        for fold in range(1, 6):
            started = perf_counter()
            outer_fit = table.fold5_kfold.to_numpy() != fold
            outer_test = ~outer_fit
            if set(groups[outer_fit]) & set(groups[outer_test]):
                raise ValueError('Group leakage in secondary outer folds')
            train_X, train_y, train_g = X[outer_fit], y[outer_fit], groups[outer_fit]
            perturbation = None
            if training_label_transform is not None:
                train_y, perturbation = training_label_transform(train_y.copy(), train_g.copy(), fold)
            inner = list(StratifiedGroupKFold(3,shuffle=True,random_state=42+fold).split(train_X,train_y,train_g))
            for fit, valid in inner:
                if set(train_g[fit]) & set(train_g[valid]):
                    raise ValueError('Group leakage in secondary inner folds')
                if set(train_y[fit]) != {0,1} or set(train_y[valid]) != {0,1}:
                    raise ValueError('Both classes required in every secondary partition')
            all_scores, candidates, rejected = [], [], []
            selection_seconds, tuning_seconds = 0., 0.
            partition_audit = []
            for config in configs:
                prepared, audit = [], []
                try:
                    for fit, valid in inner:
                        scaler, selector, transformed, indices, elapsed = _prepare_secondary(train_X[fit], train_y[fit], method, config, author_path, selector_factory)
                        selection_seconds += elapsed
                        prepared.append((transformed, train_y[fit], _transform(scaler,selector,train_X[valid]), train_y[valid]))
                        audit.append({'training_rows':len(fit),'validation_rows':len(valid),'selected_indices':indices.tolist()})
                except ZeroFeatures:
                    rejected.append(config)
                    continue
                tick = perf_counter()
                scores = Parallel(n_jobs=n_jobs,prefer='threads')(delayed(_score_candidate)(params,prepared) for params in grid)
                tuning_seconds += perf_counter()-tick
                all_scores.extend(scores)
                candidates.extend({'svm':params,'selector':config} for params in grid)
                partition_audit.append({'selector':config,'partitions':audit})
            if not candidates:
                raise ValueError('Every selector setting was ineligible on training data')
            best = choose_candidate(all_scores)
            chosen = candidates[best]
            tick = perf_counter()
            scaler, selector, transformed, indices, elapsed = _prepare_secondary(train_X,train_y,method,chosen['selector'],author_path,selector_factory)
            selection_seconds += elapsed
            params = {k.removeprefix('svm__'):v for k,v in chosen['svm'].items()}
            model = SVC(**params).fit(transformed,train_y)
            refit_seconds = perf_counter()-tick
            tick = perf_counter()
            predictions[outer_test] = model.predict(_transform(scaler,selector,X[outer_test]))
            prediction_seconds = perf_counter()-tick
            held_out_count[outer_test] += 1
            selected_rows.extend({'outer_fold':fold,'feature_name':names[i]} for i in indices)
            folds.append({'outer_fold':fold,'inner_seed':42+fold,'training_rows':int(outer_fit.sum()),
                          'test_rows':int(outer_test.sum()),'n_candidates':len(candidates),
                          'best_params':chosen['svm'],'best_selector_params':chosen['selector'],
                          'rejected_selector_settings':rejected,'inner_f1_macro':all_scores[best]['macro_f1'],
                          'inner_accuracy':all_scores[best]['accuracy'],'selector_seconds':selection_seconds,
                          'svm_tuning_seconds':tuning_seconds,'refit_seconds':refit_seconds,
                          'prediction_seconds':prediction_seconds,'total_seconds':perf_counter()-started,
                          'selected_feature_count':len(indices),'selector_partitions':partition_audit,
                          'search_results':[dict(p,**s) for p,s in zip(candidates,all_scores)],
                          'metrics':classification_metrics(y[outer_test],predictions[outer_test])})
            if perturbation is not None:
                folds[-1]['training_label_perturbation'] = perturbation
            if progress:
                progress(data.name,method,fold,len(indices),folds[-1]['total_seconds'])
    if not np.all(held_out_count==1) or not np.isin(predictions,[0,1]).all():
        raise ValueError('Every secondary case must have exactly one held-out prediction')
    prediction_table = pd.DataFrame({'image_id':table.image_id.to_numpy(),'patient_id':groups,
        'outer_fold':table.fold5_kfold.to_numpy(),'y_true':y,'y_pred':predictions})
    meta = {'method':method,'protocol_id':protocol_id,'dataset_sha256':data.sha256,'status':'COMPLETED',
            'dataset':data.name,'unit':data.unit,'seed':42,'n_jobs':n_jobs,'platform':platform.platform(),
            'python':platform.python_version(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            'selector_grid':configs,'svm_candidates_per_selector':len(grid),
            'versions':{name:version(name) for name in ('numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl')},
            'implementation_note':'Secondary benchmark; never substitute this protocol for the frozen primary evaluation.',
            'author_source_sha256':AUTHOR_SHA256 if method=='author_gbfrs' else None}
    return EvaluationResult(prediction_table,pd.DataFrame(selected_rows),folds,meta)


def aggregate_secondary(result, data):
    sets = [set(result.selected_features.loc[result.selected_features.outer_fold==f,'feature_name']) for f in range(1,6)]
    return {'method':result.metadata['method'],'n_rows':len(data.table),'n_features':len(data.feature_names),
            'unit':data.unit,'metrics':{metric:{'mean':float(np.mean([f['metrics'][metric] for f in result.folds])),
                                              'sd':float(np.std([f['metrics'][metric] for f in result.folds],ddof=1))}
                                     for metric in result.folds[0]['metrics']},
            'mean_features':float(np.mean([len(s) for s in sets])),
            'stability':{key:value for key,value in selection_stability(sets,data.feature_names).items() if key!='selection_counts'},
            'candidate_counts':[f['n_candidates'] for f in result.folds],
            'best_selector_settings':[f['best_selector_params'] for f in result.folds]}
