# Results from the complete 28 October 2026 rerun

Primary endpoint: outer-fold mean macro-F1 with sample SD. The frozen protocol was unchanged.
Detailed machine-readable aggregates are in `results/aggregate_results.json`.

## All-feature SVM

Macro-F1: 0.853359 ± 0.043567 across outer folds. Mean selected count: 288 of 288 (0.00% reduction).
Pooled OOF macro-F1: 0.854209; conditional patient-bootstrap 95% interval [0.833810, 0.874775].
Descriptive mean pairwise Jaccard: 1.0000; Nogueira stability: 1.0000.
Measured total wall time across five folds: 177.0 seconds.

## Fuzzy-rough + SVM

Macro-F1: 0.850084 ± 0.036229 across outer folds. Mean selected count: 7.6 of 288 (97.36% reduction).
Pooled OOF macro-F1: 0.850840; conditional patient-bootstrap 95% interval [0.829960, 0.871464].
Descriptive mean pairwise Jaccard: 0.2491; Nogueira stability: 0.3783.
Measured total wall time across five folds: 33.0 seconds.

## GBFRS reconstruction + SVM

Macro-F1: 0.853967 ± 0.041700 across outer folds. Mean selected count: 13 of 288 (95.49% reduction).
Pooled OOF macro-F1: 0.855053; conditional patient-bootstrap 95% interval [0.834384, 0.875772].
Descriptive mean pairwise Jaccard: 0.5313; Nogueira stability: 0.6778.
Measured total wall time across five folds: 211.2 seconds.

## Interpretation

All paired pooled-metric difference intervals span zero. No superiority, equivalence or non-inferiority conclusion follows. Methods were run sequentially, but desktop background activity was not controlled.
GBFRS is an independent paper-based reconstruction with documented adaptations. The original author archive is not redistributed. The study has no external cohort and does not establish clinical validity.
The bootstrap conditions on fixed OOF predictions. Fold SD is not a confidence interval, and named-feature stability is descriptive across overlapping training sets. See `STATISTICAL_ANALYSIS.md`.

Both the original handoff checker and the repository acceptance checker are run on the complete private submissions. Exact row-level artifacts stay private.

## Secondary extension

The original three results above remain the primary frozen comparison. See
[secondary study](SECONDARY_STUDY.md) and `results/secondary_results.json` for three
additional BUS-BRA selectors, seven WDBC pipelines and the pinned private reference
audit. WDBC is a cytology method benchmark, not external ultrasound validation.

## Controlled audit and independent ultrasound extension

Eleven robustness/sensitivity/criterion runs and six independently retrained
BrEaST ultrasound pipelines completed. See [expanded study](EXPANDED_STUDY.md),
`results/robustness_results.json` and `results/external_ultrasound_results.json`.
The primary results are unchanged; BrEaST is not frozen-model transfer testing.
