#!/usr/bin/env python3
"""Reproducible tables and label-noise figure from aggregate-only research outputs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LABELS={'baseline':'All-feature SVM','fuzzy_rough':'Fuzzy-rough + SVM','gbfrs':'GBFRS reconstruction + SVM',
        'anova':'ANOVA + SVM','mutual_information':'Mutual information + SVM','l1_logistic':'L1 logistic selection + SVM'}


def table(command,rows):
    return '\n'.join(['\\newcommand{\\'+command+'}{%',r'\begin{tabular}{lrrr}',r'\toprule',
       r'Setting & Macro-F1 (mean $\pm$ SD) & Accuracy & Features \\',r'\midrule',
       *[f"{label} & {m['metrics']['f1_macro']['mean']:.4f} $\\pm$ {m['metrics']['f1_macro']['sd']:.4f} & {m['metrics']['accuracy']['mean']:.4f} & {m['mean_features']:.1f} "+r'\\' for label,m in rows],
       r'\bottomrule',r'\end{tabular}}'])


def main():
    primary=json.loads((ROOT/'results/aggregate_results.json').read_text())
    robust=json.loads((ROOT/'results/robustness_results.json').read_text())
    external=json.loads((ROOT/'results/external_ultrasound_results.json').read_text())
    assert robust['dataset_sha256']==primary['dataset_sha256'] and robust['completed_experiments']==11
    assert external['status']=='COMPLETED' and len(external['methods'])==6
    blocks=['% Generated from validated aggregate-only records.']
    rows=[]
    for method in ('baseline','fuzzy_rough','gbfrs'):
        for rate in (0,10,20):
            m=primary['methods'][method] if rate==0 else robust['methods'][f'{method}_noise_{rate}']
            rows.append((f'{LABELS[method]}: {rate}\\%',m))
    blocks.append(table('NoiseTable',rows))
    configurations=[('Own: purity 0.90, seed 42',primary['methods']['gbfrs']),
                    ('Own: purity 0.80, seed 42',robust['methods']['gbfrs_purity_80']),
                    ('Own: purity 0.95, seed 42',robust['methods']['gbfrs_purity_95']),
                    ('Own: purity 0.90, seed 43',robust['methods']['gbfrs_seed_43']),
                    ('Own: purity 0.90, seed 44',robust['methods']['gbfrs_seed_44']),
                    ('Alternative criterion: same balls',robust['methods']['gbfrs_criterion_ablation'])]
    blocks.append(table('SensitivityTable',configurations))
    blocks.append(table('ExternalTable',[(LABELS[k],v) for k,v in external['methods'].items()]))
    (ROOT/'paper/expanded_generated.tex').write_text('\n\n'.join(blocks)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.hashsalt':'busbra-label-noise-v1'})
    fig,ax=plt.subplots(figsize=(7.1,3.6),layout='constrained')
    for method,color,marker in [('baseline','#536474','s'),('fuzzy_rough','#b75b35','^'),('gbfrs','#176c82','o')]:
        records=[primary['methods'][method],*[robust['methods'][f'{method}_noise_{rate}'] for rate in (10,20)]]
        means=[m['metrics']['f1_macro']['mean'] for m in records];sd=[m['metrics']['f1_macro']['sd'] for m in records]
        ax.errorbar([0,10,20],means,yerr=sd,color=color,marker=marker,label=LABELS[method],linewidth=1.6,capsize=3,alpha=.9)
    ax.set(xlabel='Nominal training-patient label-flip rate (%)',ylabel='Outer-fold macro-F1 (mean ± SD)',
           xticks=[0,10,20],ylim=(.67,.92),title='Training-label stress test on BUS-BRA')
    ax.grid(axis='y',color='#e5e9ed',linewidth=.7);ax.legend(frameon=False,loc='lower left',fontsize=9)
    folder=ROOT/'paper/figures'
    fig.savefig(folder/'label-noise.png',dpi=220)
    fig.savefig(folder/'label-noise.svg',metadata={'Date':None})
    svg=folder/'label-noise.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print('Generated expanded aggregate tables and label-noise figure')


if __name__=='__main__':main()
