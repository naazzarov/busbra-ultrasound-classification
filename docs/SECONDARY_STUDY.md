# Secondary study: broader baselines and implementation audit

Protocol: `evaluation/secondary_protocol.json`, ID `BUSBRA-SECONDARY-20261028-V1`.
The specification was written before examining secondary outcomes, after seeing the
primary results. It is not a public preregistration. No primary settings or scores
are replaced by this extension.

## Comparators and evaluation boundaries

ANOVA F-score and mutual-information ranking tune k in {5,10,20}. Mutual information
uses three neighbors and seed 42. L1 logistic selection tunes C in {0.01,0.1,1},
uses liblinear, seed 42, max_iter 10000, and retains coefficients above 1e-5 in
absolute value. A zero-feature inner setting is ineligible; no fallback feature is
invented. All downstream classifiers are SVMs.

Selection and scaling are refitted within each inner training partition. Three
selector settings cross the unchanged 32 SVM settings (96 joint candidates); the
original fixed selectors search 32. This is not matched-compute-budget evidence.
Outer outcomes are descriptive and never select hyperparameters. These extensions
have no confirmatory hypothesis tests or newly asserted statistical superiority.

## BUS-BRA

The input fingerprint and five supplied patient-grouped outer folds are unchanged.
Three grouped inner folds use seeds 42+outer_fold. All three additional methods
completed the entire nested experiment:

- ANOVA: mean macro-F1 0.8391, SD 0.0420; mean 11.0 predictors.
- Mutual information: mean macro-F1 0.8469, SD 0.0488; mean 9.0 predictors.
- L1 logistic selection: mean macro-F1 0.8447, SD 0.0323; mean 30.8 predictors.

The unchanged author selector also completed all five BUS-BRA outer folds: mean
macro-F1 0.5034 (SD 0.0396), accuracy 0.5730, and 1.8 mean predictors. No source
patch was applied. The run took about 815 seconds; this is uncontrolled execution
time, not a comparative speed benchmark. The outcome applies to the declared
wrapper/seed/purity/SVM setting and is not a refutation of the source paper.

The primary baseline remains 0.8534 and GBFRS reconstruction 0.8540. The difference
is small; the observed ranking is not a superiority claim.

## Independent WDBC method benchmark

Source: [UCI Wisconsin Diagnostic Breast Cancer](https://doi.org/10.24432/C5DW2B),
Wolberg, Mangasarian, N. Street and W. Street (1993), CC BY 4.0. The scikit-learn
1.6.1 bundled copy contains 569 cases and 30 cytology features from fine-needle
aspirates: 212 malignant and 357 benign. The loader explicitly inverts scikit-learn's
target encoding so malignant remains positive. The canonical matrix/labels/names
fingerprint is `a5445fe1309067463a5eca3cce2c6937633bf991d6daa5a46edf66f78fe67209`.

Five stratified outer folds use shuffling and seed 42; inner folds use the same
three-fold group splitter with one case per group. This does not assert a verified
patient mapping. All pipelines are trained afresh. WDBC is a different modality,
not a second ultrasound cohort, transfer experiment, or external validation of the
BUS-BRA classifier. No row-level copy of either dataset is published.

All seven methods completed. Mean macro-F1 / mean predictors:

- All-feature SVM: 0.9772 / 30.0.
- Conventional fuzzy-rough: 0.9455 / 5.6.
- GBFRS reconstruction: 0.9603 / 4.8.
- ANOVA: 0.9772 / 20.0.
- Mutual information: 0.9720 / 20.0.
- L1 logistic selection: 0.9680 / 14.2.
- Unchanged author selector: 0.8366 / 2.8.

The all-feature baseline exceeds both fuzzy-rough pipelines on this benchmark.
These results limit a general advantage claim, even though the selectors use fewer
inputs. Five-fold SDs and feature stability are in the aggregate JSON.

## Pinned private reference

The wrapper checks SHA-256
`8f675c052aaaab5760a853112990022bf4f5ba7afa7bb3b54d903e9467ec5c9b`
for the author-linked script at commit `0382e0899b0dc5529c89d1b9a11da198464d49ad`.
It uses training-only min-max geometry, purity 0.90 and NumPy seed 42, then calls
`attribute_reduce` unchanged. Global RNG state is restored afterward. Neither its
source nor the private archive is redistributed. This is our SVM evaluation of an
unchanged selector, not a reproduction of the authors' original classifier/tables.

The actual pinned source expression is evaluated on a synthetic three-ball fixture:
centers 0,0.2,0.9; labels 0,0,1; weights 0.6,0.3,0.1. The paper-based own-ball
criterion is 0.82, versus 0.51 for the inspected expression. A 60-row, five-feature
fixture selects three versus two features, respectively. These software fixtures
are not clinical results. The public script reads the private expression through
its AST; no third-party expression or implementation is vendored.

## Validation and remaining limits

`tools/summarize_secondary.py` validates private row/group/fold/label alignment,
unique and complete held-out predictions, allowed selected columns, recomputed fold
metrics, candidate counts and source-snapshot hashes. Only aggregate quantities
are exported. Timing is not presented as an isolated speed benchmark.

The later BrEaST extension adds independent ultrasound method evidence (see
EXPANDED_STUDY.md). Frozen-model external testing, a broader dataset suite, verified author
contributions/affiliations and human scientific review remain outstanding. This
extension improves the evidence; it cannot guarantee publication or a score of
10/10. No arXiv submission has been made.
