#!/usr/bin/env python3
"""Audit a pinned private reference without copying its source into this repo."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from busbra.secondary import AUTHOR_SHA256, _load_author_module
from busbra.selectors import GBFRSSelector, ball_dependency


def audit(path):
    path=Path(path)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=AUTHOR_SHA256:
        raise ValueError('Reference source hash mismatch')
    tree=ast.parse(raw.decode('utf-8-sig'))
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='attribute_reduce')
    expression=next(node.value for node in ast.walk(function) if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='sum_each_ball' for t in node.targets)
        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='sum')
    centers=np.array([[0.],[.2],[.9]])
    labels=np.array([0,0,1]); weights=np.array([.6,.3,.1])
    distances=abs(centers-centers.T)
    nearest,nearest_weights=[],[]
    for label in (0,1):
        filtered=np.where(labels[None,:]!=label,distances,np.inf)
        nearest.append(filtered.min(axis=1))
        nearest_weights.append(weights[filtered.argmin(axis=1)])
    env={'__builtins__':{'sum':sum,'range':range,'len':len},'np':np,
         'nearest_distances':nearest,'w_nearest_distances':np.asarray(nearest_weights)}
    author=float(eval(compile(ast.Expression(expression),'pinned-author-expression','eval'),env,env))
    paper=ball_dependency(centers,labels,weights,[0],1)
    assert np.isclose(paper,.82) and np.isclose(author,.51)
    module=_load_author_module(str(path.resolve()))
    rng=np.random.RandomState(10)
    X=np.r_[rng.normal(.1,.01,(30,5)),rng.normal(.8,.01,(30,5))]
    X=(X-X.min(0))/np.ptp(X,axis=0); y=np.repeat([0,1],30)
    state=np.random.get_state()
    try:
        with threadpool_limits(1),contextlib.redirect_stdout(io.StringIO()):
            np.random.seed(42)
            selected=[int(i) for i in module.attribute_reduce(np.column_stack([X,y,np.arange(60)]),pur=.9)]
            reconstructed=GBFRSSelector().fit(X,y).get_support(indices=True).tolist()
    finally:
        np.random.set_state(state)
    return {'reference_commit':'0382e0899b0dc5529c89d1b9a11da198464d49ad','source_sha256':AUTHOR_SHA256,
            'source_redistributed':False,'dependency_fixture':{'synthetic':True,'centers':[0.,.2,.9],
                'ball_labels':[0,0,1],'ball_weights':[.6,.3,.1],'paper_own_ball_weighted':paper,
                'unchanged_author_expression':author,'numerically_equivalent':False},
            'selector_fixture':{'synthetic':True,'n_rows':60,'n_features':5,'purity':.9,'seed':42,
                'author_selected_indices':selected,'reconstruction_selected_indices':reconstructed},
            'scope':'Algebraic and software fixtures only. Separate BUS-BRA and WDBC nested SVM runs test the unchanged private reference. No claim to reproduce original classifier or accuracy tables.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--author-code',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.author_code)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Reference audit written:',args.output)
