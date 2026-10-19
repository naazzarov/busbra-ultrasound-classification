# BUS-BRA Granular Computing — Breast Tumour Classification

## Personal portfolio extension — in progress

This personal extension targets `naazzarov/BUS-BRA-Granular-Computing` and retains
the original team project's history and specifications. See
[development progress and runnable commands](docs/PORTFOLIO_PROGRESS.md).

The shared data validator and full nested baseline SVM workflow are implemented
and tested on synthetic fixtures. The fuzzy-rough and GBFRS implementations,
official submission export, and real experiments are pending the private team
handoff and verified method references. No BUS-BRA performance results are
claimed. The dataset and row-level predictions stay private.

The original team brief follows.

**A three-person university project:** binary benign/malignant classification comparing an all-feature SVM, conventional fuzzy-rough feature selection + SVM, and published granular-ball fuzzy-rough feature selection (GBFRS) + SVM.

**Status:** repository ready for development. No model results are claimed yet.

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
1. `git clone https://github.com/Granular-Breast-Cancer-Team/BUS-BRA-Granular-Computing.git`
2. Use Python 3.11 and install your method's dependencies. Record exact package versions.
3. Copy the already distributed CSV into `inputs/BUSBRA_features_ready.csv`. Confirm its SHA-256 before training; do **not commit** it.
4. Work only on your assigned `feature/*` branch and method folder.
5. Push commits to your branch and open a pull request to main. Coordinator reviews before merging.

**Result comparison:** the complete, tested `tools/compare_results.py` and submission templates are already in the **BUSBRA_Team_Handoff.zip** shared privately with the team. Continue using that original package's schema; avoid incompatible new formats. Return submitted predictions to the coordinator privately until derived-data publication rights are confirmed.

## Source and attribution
[Gómez-Flores, Gregorio-Calas & Pereira, BUS-BRA, *Medical Physics* (2024)](https://doi.org/10.1002/mp.16812). Official dataset: https://zenodo.org/records/8231412. Source `LICENSE.txt` must be checked before public data redistribution. Educational research only; not a medical diagnostic system.
