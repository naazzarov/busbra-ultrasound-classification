# Breast Ultrasound Classification with Fuzzy-Rough Feature Selection

[![Python tests](https://github.com/naazzarov/busbra-ultrasound-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/naazzarov/busbra-ultrasound-classification/actions/workflows/tests.yml)
![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue)

A controlled comparison of **all-feature SVM**, **conventional fuzzy-rough selection + SVM**,
and **a documented GBFRS reconstruction + SVM** on BUS-BRA radiomics. The question is
whether the classifier can use far fewer measurements while retaining similar
observed performance under patient-grouped nested evaluation.

**Status:** all three complete real-data runs reproduced on 28 October 2026.
Three additional selectors and a separate seven-method cytology benchmark extend the study.
Eleven robustness/ablation runs and six pipelines on an independent ultrasound
cohort are also complete. The paper remains a research draft for author review.
The dataset, identifiers, row-level predictions and model artifacts remain private.

## Results at a glance

The unchanged input contains **1,875 images**, **1,064 patient groups** and
**288 named predictors**. Every method uses the same five supplied outer folds,
three grouped inner folds and 32 SVM candidates.

| Method | Outer-fold macro-F1, mean ± SD | Mean features | Feature reduction |
|---|---:|---:|---:|
| All-feature SVM | 0.8534 ± 0.0436 | 288 | 0.00% |
| Fuzzy-rough + SVM | 0.8501 ± 0.0362 | 7.6 | 97.36% |
| GBFRS reconstruction + SVM | 0.8540 ± 0.0417 | 13 | 95.49% |

![Held-out performance and classifier input counts](paper/figures/performance-reduction.svg)

The point differences are small. All paired conditional patient-bootstrap intervals
for differences in pooled macro-F1 span zero. This does **not** establish equivalence,
non-inferiority or clinical utility. The main observation is a substantial reduction
in classifier inputs with close observed scores on this dataset.

[Aggregate results](results/aggregate_results.json) · [Statistical interpretation](docs/STATISTICAL_ANALYSIS.md) · [Paper source](paper/main.tex)

## Implementation audit and independent ultrasound evidence

**31 dataset/method/settings runs and 155 outer evaluations are complete.**
The extra runs are controlled analyses, not independent replications of BUS-BRA.

- Changing only the GBFRS dependency expression, while keeping ball construction
  fixed, changes macro-F1 from **0.8540 to 0.5051** and mean selected counts from
  13.0 to 1.8. This isolates a substantial implementation effect.
- With nominal 10% and 20% training-patient label corruption, GBFRS gives macro-F1
  **0.8405 and 0.8181**. These are descriptive stress draws, not a robustness guarantee.
- Six pipelines completed on **252 BrEaST ultrasound patients**, with a public,
  separately specified 102-feature extraction. ANOVA scores **0.7296**, L1 selection
  **0.7272**, and GBFRS **0.6700**. The evidence does not show a universal GBFRS advantage.

BrEaST models are trained afresh: this is independent-cohort method evaluation,
not external testing of a frozen BUS-BRA model. The missing original extraction
configuration still prevents that exact transfer experiment. No private handoff
or author source was sent to the public-data runner.

[Expanded study](docs/EXPANDED_STUDY.md) · [Robustness aggregates](results/robustness_results.json) ·
[Independent ultrasound aggregates](results/external_ultrasound_results.json) ·
[Successful benchmark run](https://github.com/naazzarov/busbra-ultrasound-classification/actions/runs/40668315059)

![Training-label stress test; error bars show fold SD](paper/figures/label-noise.svg)

## Reproduce the evaluation

Use Python 3.12 and the frozen private team CSV. The CLI verifies its byte fingerprint
before training. The raw image dataset alone does not reproduce the prepared table
without the upstream extraction environment.

```bash
git clone https://github.com/naazzarov/busbra-ultrasound-classification.git
cd busbra-ultrasound-classification
uv sync --locked
uv run --locked pytest
uv run --locked busbra validate --data /PRIVATE/PATH/BUSBRA_features_ready.csv
uv run --locked busbra run-all \
  --data /PRIVATE/PATH/BUSBRA_features_ready.csv \
  --output private_outputs/new-run --jobs 2
uv run --locked busbra summarize \
  --data /PRIVATE/PATH/BUSBRA_features_ready.csv \
  --submissions private_outputs/new-run/submissions \
  --output private_outputs/new-run/aggregate_results.json --bootstrap 5000
```

`run-all` executes methods sequentially and records completion or failure in a private
manifest. It refuses to overwrite an existing run. Each method also has its own CLI
command: `baseline`, `fuzzy_rough` or `gbfrs` with the same data/output/jobs arguments.
There are no fabricated sample outputs or automatic downloads of research data.

## How information stays inside training partitions

```mermaid
flowchart TD
    A[Frozen CSV: validate fingerprint and schema] --> B[Supplied outer patient-grouped fold]
    B --> C[Three patient-grouped inner splits]
    C --> D[Fit scaler and selector on inner training only]
    D --> E[Search 32 SVM candidates on inner validation]
    E --> F[Refit fresh transforms and chosen SVM on outer training]
    F --> G[Predict outer test once]
    G --> H[Private full audit and aggregate-only publication]
```

Feature selection is refitted on every inner training set and outer refit. No
clinical metadata, engineered predictors, PCA, global filtering, resampling or
test-tuned decision threshold is introduced. All 288 named predictors remain
available, including the 42 reported exact-value redundancies.

## Extended experiments and reference audit

The separately specified secondary study adds **ANOVA**, **mutual information** and
**L1 logistic feature selection**, each followed by the same SVM grid. Their BUS-BRA
mean macro-F1 scores are 0.8391, 0.8469 and 0.8447, with 11.0, 9.0 and 30.8 mean
predictors. These selectors tune three settings inside the inner folds, giving 96
joint candidates versus 32 for the fixed primary selectors; compute budgets differ.

A separate **WDBC cytology benchmark** trains seven pipelines from scratch on 569
cases and 30 predictors. The all-feature SVM achieves mean macro-F1 0.9772, compared
with 0.9603 for the GBFRS reconstruction and 0.9455 for conventional fuzzy-rough
selection. Thus neither fuzzy-rough method is uniformly preferable. WDBC is a
method benchmark, **not external validation of an ultrasound model**.

The pinned unchanged author selector completes both datasets: BUS-BRA macro-F1
0.5034 with 1.8 mean predictors; WDBC macro-F1 0.8366 with 2.8 mean predictors.
These outcomes apply to the declared wrapper and SVM evaluation setting. A synthetic criterion audit gives 0.82 for own-ball weighting
and 0.51 for the inspected author expression. These implementations are explicitly
not interchangeable. No third-party author source is distributed.

[Secondary study](docs/SECONDARY_STUDY.md) · [Protocol](evaluation/secondary_protocol.json) ·
[Validated aggregate results](results/secondary_results.json) · [Reference fixture](results/reference_audit.json)

```bash
uv run --locked python tools/run_secondary.py --dataset wdbc --output private_outputs/wdbc-new --jobs 2
uv run --locked python tools/run_secondary.py --dataset busbra --data /PRIVATE/PATH/BUSBRA_features_ready.csv --output private_outputs/baselines-new --jobs 2
# Optional private reference: add --author-code /PRIVATE/PATH/GBFRS.py and --methods author_gbfrs
```

The runner records failures and wall-clock limits; it never substitutes a patched
reference or reports partial runs as complete. Human author review and frozen-model
external ultrasound testing remain outstanding.

## GBFRS implementation qualification

The GBFRS pipeline is an **independent reconstruction of the accessible 2025
preprint**, with documented normalization, initialization, overlap handling and
stopping choices. It uses the paper's own-ball-weighted dependency in Equation 13.
The inspected author archive has no code license and computes a different dependency
expression. The primary pipeline does not import it. An optional, hash-pinned private reference runner is now available; the author source is never redistributed. These results are not a
numerical replication of the author program or its published benchmark tables.

[Method equations](docs/METHODS.md) · [Reference audit](docs/REFERENCE_AUDIT.md) · [Evaluation contract](evaluation/evaluation_contract.json)

## Paper and analysis

The draft is **Patient-Grouped Evaluation of Fuzzy-Rough Feature Selection for Breast
Ultrasound Classification**, by Bahodir Nazarov, Zineb Gharib and Fatima Zazoul;
Luo Chuan is acknowledged. Author details and research statements require author
review before submission.

- [Manuscript and build instructions](paper/README.md)
- [Results and uncertainty](docs/RESULTS.md)
- [Dataset provenance and restrictions](docs/DATASET.md)
- [Exact private submission format](docs/SUBMISSION_FORMAT.md)
- [Development and completion record](docs/PORTFOLIO_PROGRESS.md)

The public JSON contains aggregate metrics only. The conditional bootstrap holds
OOF predictions fixed and does not estimate all retraining uncertainty. Stability
is descriptive across five overlapping training sets. BrEaST adds independently
retrained ultrasound method evidence. Frozen-model external testing and clinical
validation remain outstanding; no new feature-selection algorithm is claimed.

## Source, attribution and reuse

This personal extension retains the original team's Git history and credits.
Historical team assignments remain documented in [TEAM_ROLES.md](docs/TEAM_ROLES.md).
Use [CITATION.cff](CITATION.cff) to cite the software and cite the original research:

- Gómez-Flores, Gregorio-Calas and Pereira, [BUS-BRA](https://doi.org/10.1002/mp.16812), *Medical Physics* (2024).
- Jensen and Shen, [fuzzy-rough feature selection](https://doi.org/10.1109/TFUZZ.2008.924209), *IEEE TFS* (2009).
- Xia et al., [GBFRS preprint](https://arxiv.org/abs/2501.18413) (2025).

Original images are distributed by their creators through [Zenodo](https://zenodo.org/records/8231412).
Dataset availability does not authorize redistribution of this private handoff.
No blanket software or manuscript reuse license is asserted on behalf of other
contributors. This is computational research, not a medical diagnostic service.
