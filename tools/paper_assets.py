#!/usr/bin/env python3
"""Build publication figures/tables from the aggregate-only JSON artifact.

Optional figure environment: python -m pip install matplotlib==3.10.8
No dataset, identifiers, or row-level predictions are read by this script.
"""
import argparse
import json
from pathlib import Path

NAMES = {'baseline':'All-feature SVM','fuzzy_rough':'Fuzzy-rough + SVM','gbfrs':'GBFRS reconstruction + SVM'}
SHORT = {'baseline':'All features','fuzzy_rough':'Fuzzy-rough','gbfrs':'GBFRS reconstruction'}


def build(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    (output/'figures').mkdir(parents=True, exist_ok=True)
    methods = summary['methods']
    order = list(NAMES)
    plt.rcParams.update({'font.size':11.5,'axes.spines.top':False,'axes.spines.right':False,
                         'font.family':'DejaVu Sans','savefig.bbox':'tight','svg.fonttype':'none'})
    fig, axes = plt.subplots(1,2,figsize=(9,3.2),gridspec_kw={'width_ratios':[1.25,1]})
    for i,m in enumerate(order):
        score=methods[m]['metrics']['f1_macro']
        axes[0].errorbar(score['mean'],i,xerr=score['sd'],fmt='o',color='#245b73',capsize=4)
    axes[0].set(yticks=range(3),yticklabels=[SHORT[m] for m in order],xlabel='Outer-fold macro-F1 (mean ± sample SD)',xlim=(.75,.94))
    axes[0].invert_yaxis()
    axes[0].grid(axis='x',color='.88',linewidth=.6)
    axes[0].set_title('(a) Held-out classification')
    axes[1].barh(range(3),[methods[m]['mean_features'] for m in order],color='#245b73')
    axes[1].set(yticks=range(3),yticklabels=[],xlabel='Mean selected features (of 288)',xlim=(0,330))
    axes[1].invert_yaxis()
    axes[1].set_title('(b) Predictor count')
    for i,m in enumerate(order):
        value=methods[m]['mean_features']
        axes[1].text(value+4,i,f'{value:g}',va='center')
    fig.tight_layout()
    for suffix in ('png','svg'):
        fig.savefig(output/'figures'/f'performance-reduction.{suffix}',dpi=220)
    plt.close(fig)
    labels,values,low,high=[],[],[],[]
    for key,row in summary['paired_effects'].items():
        second,first=key.split('_minus_')
        labels.append(f'{SHORT[second]} − {SHORT[first]}')
        values.append(row['pooled_macro_f1_difference'])
        low.append(row['patient_bootstrap_95ci'][0]); high.append(row['patient_bootstrap_95ci'][1])
    fig, ax=plt.subplots(figsize=(8.8,3.3))
    ax.axvline(0,color='.4',linestyle='--',linewidth=1)
    for i,(v,l,h) in enumerate(zip(values,low,high)):
        ax.plot([l,h],[i,i],color='#245b73',linewidth=2)
        ax.plot(v,i,'o',color='#245b73')
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Difference in pooled OOF macro-F1\n95% patient-bootstrap interval')
    ax.set_title('Paired comparison of fixed out-of-fold predictions',fontsize=11.5)
    ax.invert_yaxis(); ax.grid(axis='x',color='.88',linewidth=.6)
    fig.tight_layout()
    for suffix in ('png','svg'):
        fig.savefig(output/'figures'/f'paired-effects.{suffix}',dpi=220)
    plt.close(fig)
    tex=['% Generated from results/aggregate_results.json. Do not hand-edit.']
    macros={'baseline':'Baseline','fuzzy_rough':'Fuzzy','gbfrs':'GB'}
    for method,prefix in macros.items():
        row=methods[method]
        for key,metric in [('Fone','f1_macro'),('Accuracy','accuracy'),('Recall','malignant_sensitivity'),('Specificity','benign_specificity')]:
            tex.append('\\newcommand{\\'+prefix+key+'}{'+f"{row['metrics'][metric]['mean']:.4f}"+'}')
        tex.append('\\newcommand{\\'+prefix+'Features}{'+f"{row['mean_features']:g}"+'}')
        tex.append('\\newcommand{\\'+prefix+'Reduction}{'+f"{row['feature_reduction_pct']:.2f}"+'}')
    tex += ['\\newcommand{\\PrimaryTable}{%', '\\begin{tabular}{lrrrr}\\toprule',
            'Method & Macro-F1 & Accuracy & Features & Reduction (\\%)\\\\\\midrule']
    for m in order:
        row=methods[m]; f=row['metrics']['f1_macro']; a=row['metrics']['accuracy']
        tex.append(f"{SHORT[m]} & {f['mean']:.4f} $\\pm$ {f['sd']:.4f} & {a['mean']:.4f} $\\pm$ {a['sd']:.4f} & {row['mean_features']:g} & {row['feature_reduction_pct']:.2f} \\\\")
    tex += ['\\bottomrule\\end{tabular}}', '\\newcommand{\\ClinicalTable}{%',
            '\\begin{tabular}{lrrr}\\toprule','Method & Balanced accuracy & Malignant recall & Benign specificity\\\\\\midrule']
    for m in order:
        row=methods[m]
        text=' & '.join(f"{row['metrics'][k]['mean']:.4f} $\\pm$ {row['metrics'][k]['sd']:.4f}" for k in ('balanced_accuracy','malignant_sensitivity','benign_specificity'))
        tex.append(f'{SHORT[m]} & {text} \\\\')
    tex += ['\\bottomrule\\end{tabular}}','\\newcommand{\\StabilityTable}{%',
            '\\begin{tabular}{lrrr}\\toprule','Method & Mean Jaccard & Nogueira stability & Total time (s)\\\\\\midrule']
    for m in order:
        row=methods[m]
        tex.append(f"{SHORT[m]} & {row['stability']['mean_pairwise_jaccard']:.3f} & {row['stability']['nogueira']:.3f} & {row['total_wall_seconds']:.1f} \\\\")
    tex += ['\\bottomrule\\end{tabular}}','\\newcommand{\\EffectTable}{%',
            '\\begin{tabular}{lrr}\\toprule','Comparison & Pooled macro-F1 difference & 95\\% interval\\\\\\midrule']
    for key,row in summary['paired_effects'].items():
        second,first=key.split('_minus_'); l,h=row['patient_bootstrap_95ci']
        tex.append(f"{SHORT[second]} minus {SHORT[first]} & {row['pooled_macro_f1_difference']:+.4f} & [{l:+.4f}, {h:+.4f}] \\\\")
    tex += ['\\bottomrule\\end{tabular}}']
    (output/'generated.tex').write_text('\n'.join(tex)+'\n')
    for svg in (output/'figures').glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=Path('results/aggregate_results.json'))
    parser.add_argument('--output',type=Path,default=Path('paper'))
    args=parser.parse_args()
    build(json.loads(args.summary.read_text()),args.output)
