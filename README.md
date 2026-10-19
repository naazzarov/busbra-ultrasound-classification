# BUS-BRA Granular Computing — Breast Tumour Classification

## Personal portfolio extension

This extension implements all three methods and retains the original team's
history and attribution. See [progress and runnable commands](docs/PORTFOLIO_PROGRESS.md),
[method definitions](docs/METHODS.md) and [reference verification](docs/REFERENCE_AUDIT.md).
The common runner fits preprocessing and selectors inside patient-grouped nested
cross-validation and exports the exact private handoff format. Equation, leakage,
privacy and export checks run in CI. Real-data experiments and scores remain local.

GBFRS is an independent paper reconstruction with documented adaptations. The
author archive is unlicensed and differs from the paper's weighting equation;
our source implements Equation 13 and does not redistribute the archive.

The original team brief follows.

**A three-person university project:** binary benign/malignant classification comparing an all-feature SVM, conventional fuzzy-rough feature selection + SVM, and published granular-ball fuzzy-rough feature selection (GBFRS) + SVM.

**Status:** all three implementations available. Research results are private.

## Dataset — frozen for all three methods
The verified BUS-BRA radiomics table has **1,875 image records**, **1,064 patient groups**, and **288 named numerical features** (102 original-image + 186 wavelet-derived). Of these, 42 columns are exact redundancies; all **288 must remain available at the start of every method**. Labels: 0 = benign, 1 = malignant.

**Frozen CSV:** `inputs/BUSBRA_features_ready.csv` (kept locally, **not published** here until rights are confirmed). SHA-256: `7f22dccf4be1f8920a0e9fbd46ae31ed814f1be138e55dc936b00b76d67324ef`. Read [dataset documentation](docs/DATASET.md).

## Who does what?
| Member | Implementation | Folder | Branch |
|---|---|---|---|
| **Coordinator** | **GBFRS + SVM**, final integration and comparison | [gbfrs](gbfrs) | `feature/gbfrs` |
| **Member 2** | Conventional fuzzy-rough + SVM | [fuzzy_rough](fuzzy_rough) | `feature/fuzzy-rough` |
| **Member 3** | All-feature SVM | [baseline_svm](baseline_svm) | `feature/svm-baseline` |

**These current roles supersede the older ZIP/PDF role assignments.** See [team roles](docs/TEAM_ROLES.md).

## Fair experiment — same rules for everybody
- Five predefined **patient-disjoint** held-out test folds: `fold5_kfold`, 1–5.
- In each outer training set, grouped and stratified **3-fold** inner cross-validation, seed `42 + outer_fold`.
- Tune using **macro-F1**, with the same frozen [SVM grid](evaluation/evaluation_contract.json): C ∈ {0.1, 1, 10, 100}; linear or RBF; class weights None/balanced; RBF gamma scale/0.01/0.1.
- Fit StandardScaler and any feature selector only within training folds. No extra predictors, PCA, leakage, or post-hoc test-score tuning.
- Submit exact five-fold held-out predictions (all **1,875** images), per-fold selected features, parameters, measured timings, reproducible code and software versions.
- Compare accuracy, macro-F1, balanced accuracy, malignant sensitivity, benign specificity, feature reduction, and runtimes. No accuracy level is guaranteed.

Read [experimental protocol](docs/EVALUATION_PROTOCOL.md), [submission requirements](docs/SUBMISSION_FORMAT.md), and [contribution workflow](CONTRIBUTING.md).

## How to start and collaborate
1. `git clone https://github.com/naazzarov/BUS-BRA-Granular-Computing.git`
2. Use Python 3.12 and `uv sync --locked`. Record exact package versions.
3. Copy the already distributed CSV into `inputs/BUSBRA_features_ready.csv`. Confirm its SHA-256 before training; do **not commit** it.
4. Work only on your assigned `feature/*` branch and method folder.
5. Push commits to your branch and open a pull request to main. Coordinator reviews before merging.

**Result comparison:** `tools/compare_results.py --root PRIVATE_HANDOFF_ROOT` validates the three private submissions against the frozen CSV. The runner exports the original handoff schema. Return submissions privately; keep all source and derived data out of GitHub.

## Source and attribution
[Gómez-Flores, Gregorio-Calas & Pereira, BUS-BRA, *Medical Physics* (2024)](https://doi.org/10.1002/mp.16812). Official dataset: https://zenodo.org/records/8231412. Source `LICENSE.txt` must be checked before public data redistribution. Educational research only; not a medical diagnostic system.
