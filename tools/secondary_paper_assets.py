#!/usr/bin/env python3
"""Generate secondary-study LaTeX tables from validated aggregate JSON only."""
import json
from pathlib import Path

LABELS={'baseline':'All-feature SVM','fuzzy_rough':'Fuzzy-rough + SVM','gbfrs':'GBFRS reconstruction + SVM',
        'anova':'ANOVA + SVM','mutual_information':'Mutual information + SVM',
        'l1_logistic':'L1 logistic selection + SVM','author_gbfrs':'Unchanged author selector + SVM'}


def render(payload, primary):
    if payload["datasets"]["busbra"]["dataset_sha256"] != primary["dataset_sha256"]:
        raise ValueError("Primary and secondary BUS-BRA fingerprints differ")
    tables=[]
    for dataset, command in [('busbra','BusSecondaryTable'),('wdbc','WdbcTable')]:
        methods=dict(primary['methods']) if dataset=='busbra' else {}
        methods.update(payload['datasets'][dataset]['methods'])
        lines=['\\newcommand{\\'+command+'}{%',r'\begin{tabular}{lrrr}',r'\toprule',
               r'Method & Macro-F1 (mean $\pm$ SD) & Accuracy & Features \\',r'\midrule']
        for method,label in LABELS.items():
            if method not in methods:continue
            m=methods[method]; f=m['metrics']['f1_macro'];a=m['metrics']['accuracy']['mean']
            lines.append(f"{label} & {f['mean']:.4f} $\\pm$ {f['sd']:.4f} & {a:.4f} & {m['mean_features']:.1f} "+r'\\')
        lines.extend([r'\bottomrule',r'\end{tabular}}'])
        tables.append('\n'.join(lines))
    return '% Generated from validated aggregate-only secondary results.\n'+'\n\n'.join(tables)+'\n'


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    content=render(json.loads((root/'results/secondary_results.json').read_text()),
                   json.loads((root/'results/aggregate_results.json').read_text()))
    (root/'paper/secondary_generated.tex').write_text(content)
    print('Generated paper/secondary_generated.tex')
