# Controlled implementation audit, robustness and independent ultrasound cohort

All eleven robustness runs completed and passed private export validation. Six
additional pipelines completed on the independent BrEaST ultrasound cohort.
Together with the earlier fourteen dataset/method runs, the release now contains
31 completed dataset/method/settings runs and 155 outer-fold evaluations.
These include repeated settings on the same BUS-BRA cohort; they are not 31
independent scientific replications.

## What the new evidence establishes

The controlled GBFRS ablation retains ball construction, min-max geometry, purity
0.90, seed 42, nesting, SVM grid and stopping convention. Only the dependency
criterion changes. Mean macro-F1 changes from 0.8540 with 13.0 predictors to 0.5051
with 1.8. The inspected author expression is independently implemented as the sum
of classwise maximum distances times maximum nearest-ball weights, divided by
sqrt(number of selected features). This localizes a substantial criterion effect
within our reconstruction. It is not an exact reproduction of the complete author
program, a new algorithm, or a refutation of its original publication.

Four other GBFRS settings (purity 0.80/0.95 with seed 42, and seeds 43/44 with
purity 0.90) give mean macro-F1 0.8504–0.8537, versus the primary 0.8540. They are
sensitivity analyses, not outer-fold selection of a new best configuration.

## Patient-label corruption stress test

Within each outer training set, a nominal 10% or 20% of patient groups in each
original diagnosis are selected and their labels flipped. All images of a patient
change together. The same corrupted labels enter inner fitting and validation,
and corrupted labels determine the inner stratification. Outer test labels stay
unchanged. All three methods share the draw at a given outer fold/rate.

- Nominal 10%: all-feature macro-F1 0.8299; fuzzy-rough 0.7898; GBFRS 0.8405.
- Nominal 20%: all-feature 0.8094; fuzzy-rough 0.7548; GBFRS 0.8181.

All deteriorate relative to clean runs. GBFRS has the highest observed mean in
these particular stress draws, but one draw per fold/rate, no multiplicity-adjusted
confirmatory tests and overlapping training folds limit interpretation. This is
not a guarantee of robustness to other label noise or image corruption.

The immutable primary experiment is unchanged. The additional plan was specified
before these new outcomes, after the earlier results had been examined; it is not
public preregistration. No setting is chosen using its outer-fold outcome.

## Independent ultrasound cohort: BrEaST

Sources: [TCIA data release](https://doi.org/10.7937/9WKK-Q141) and
[Pawlowska et al., Scientific Data (2024)](https://doi.org/10.1038/s41597-024-02984-z),
CC BY 4.0. There are 256 source cases from distinct patients in Poland. The four
normal scans are excluded from the binary lesion task; 154 benign and 98 malignant
cases remain. Clinical source labels are verified by biopsy or follow-up, rather
than every benign case being asserted biopsy-proven.

A separately specified extractor computes 102 original-image PyRadiomics 3.0.1
features with binWidth=16, unit pixel spacing, no intensity normalization, no
resampling/resizing, grayscale images and the nonzero tumor-mask union. Provided
segmentation defines the feature ROI; PyRadiomics' usual internal ROI processing
is retained. Multiple lesion components are not excluded. No clinical descriptors,
BI-RADS, interpretation or histology are predictors. The input and protocol hashes,
versions, zero extraction failures and zero duplicate grayscale pixel hashes within
this cohort are recorded in the public aggregate file.

All six methods use five stratified patient-disjoint outer folds and three inner
folds, with the previously specified SVM/selector grids. The models are retrained
on this cohort; no BUS-BRA weights or tuned settings are transferred. Mean macro-F1
and mean features are:

- All-feature SVM: 0.6886 / 102.0.
- Fuzzy-rough: 0.6737 / 5.8.
- GBFRS reconstruction: 0.6700 / 12.2.
- ANOVA: 0.7296 / 18.0.
- Mutual information: 0.7172 / 18.0.
- L1 logistic selection: 0.7272 / 9.6.

This does not show a universal GBFRS advantage. It supplies independent ultrasound
method evidence with a different feature basis; it is **not external validation
of a frozen BUS-BRA-trained model**. The original extraction configuration and
BUS-BRA source images are absent from the handoff, so exact upstream reproduction,
frozen-model transfer and cross-cohort pixel-duplicate checking remain unavailable.

## Execution, validation and data protection

The first remote attempt stopped before training because the released spreadsheet
spells a column differently from the published description. The parser now
normalizes case/spacing/punctuation without changing column meaning or diagnosis.
The corrected [benchmark run](https://github.com/naazzarov/busbra-ultrasound-classification/actions/runs/40668315059)
completed successfully. Only public, licensed BrEaST files were downloaded to its
ephemeral runner. The private handoff and private author source were never sent.
Only aggregate measurements were uploaded; row-level runner files were temporary.

`tools/summarize_robustness.py` checks source snapshots, complete OOF alignment,
recomputed metrics, allowed features/candidate counts and deterministic label-flip
audits against private outputs. The remote external runner checks source coverage,
patient uniqueness, finite extraction, nesting and OOF alignment/recomputed metrics.
Its source commit and execution URL are pinned in the aggregate provenance.

Public artifacts:

- `evaluation/robustness_protocol.json` and `results/robustness_results.json`.
- `evaluation/external_ultrasound_protocol.json` and `results/external_ultrasound_results.json`.
- `tools/expanded_paper_assets.py` for the three additional tables and noise plot.

Human author review, contributions/affiliations and frozen-model external testing
remain outstanding. There is no arXiv submission, accepted-paper claim, novel
algorithm claim or guaranteed numeric project/paper rating.

## Fresh-run reproducibility check

The final-source [repeat run](https://github.com/naazzarov/busbra-ultrasound-classification/actions/runs/40671134242)
also completed successfully. All reported model aggregates, including mean/SD
performance, feature counts, stability and chosen selector settings, match the
first successful run. Source-archive checksums, extraction versions and protocol
checksum also match. The extracted feature CSV has a different byte checksum
across the fresh runners; its cause has not been established by a row-level
numerical comparison. Therefore bitwise extraction identity and exact individual
prediction reproduction across those runners are not asserted. Each run records
its own feature-table hash. This is a software re-execution check, not an
additional independent clinical cohort or new scientific replication.
