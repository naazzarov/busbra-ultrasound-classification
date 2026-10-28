# Personal portfolio development

Target repository: `naazzarov/busbra-ultrasound-classification`.

This extension preserves the original team's history and specifications from
`Granular-Breast-Cancer-Team/BUS-BRA-Granular-Computing` at commit
`c01ab91863b9fdd25cae9ae859a7fd2b00f66623`. Team attribution remains intact.

## Implemented

- Immutable CSV fingerprint and structural validation.
- Three executable methods: all-feature SVM, conventional fuzzy-rough
  QuickReduct + SVM, and paper-defined GBFRS + SVM.
- The exact 32-candidate common SVM search, five predefined patient-disjoint
  outer folds and three stratified patient-grouped inner folds.
- Training-only scaling and selection, with explicit metric/accuracy/feature
  count/grid-order tie handling.
- Exact handoff submission exports, complete inner-search audit records,
  selected names, measured timings, runnable source and software versions.
- Original comparison checker restored, with optional private-root argument
  and a pinned input-fingerprint check.
- Synthetic equation checks, paper-example reproduction, sample-coverage and
  overlap tests, actual scaler/selector training-boundary audits, output-schema
  tests and checks that private artifacts are ignored.
- Defense-in-depth data ignore rules and CI checks of tracked file contents.

## Completed private validation — 2026-10-19

All three real-data experiments completed the five fixed outer folds, producing
1,875 held-out predictions per method. The unchanged original handoff checker
and repository checker both validated all three submissions. All 31 local tests
and both GitHub CI runs passed for the implementation commit. Complete private
submissions were returned to the original handoff folders; detailed scores,
source checksums and completion manifest remain only in local outputs.

GBFRS retains the independent-reconstruction qualification below. Completion of
the software and locked evaluation does not establish statistical superiority
or clinical validity. Concurrent method processes make timings descriptive,
not isolated speed benchmarks.

## Reference verification

The four-page private handoff PDF and frozen dataset are now available locally.
See [method definitions](METHODS.md) and [reference audit](REFERENCE_AUDIT.md).
The author-linked GBFRS archive has no code license and its dependency expression
differs from paper Eq. 13. Our independent reconstruction implements the paper's
own-ball weighting, with documented deterministic construction and stopping
choices. It is not an unmodified execution of the authors' code and does not
claim to reproduce their benchmark scores or journal experiments.

## Development and private experiments

Use Python 3.12 (3.11 also supported) and the checked-in lockfile:

```bash
uv sync --locked
uv run --locked pytest
uv run --locked python tools/check_public_files.py
uv run --locked busbra validate --data /PRIVATE/HANDOFF/inputs/BUSBRA_features_ready.csv
uv run --locked busbra baseline --data /PRIVATE/HANDOFF/inputs/BUSBRA_features_ready.csv --output private_outputs/run-001/submissions/baseline --jobs 2
uv run --locked busbra fuzzy_rough --data /PRIVATE/HANDOFF/inputs/BUSBRA_features_ready.csv --output private_outputs/run-001/submissions/fuzzy_rough --jobs 2
uv run --locked busbra gbfrs --data /PRIVATE/HANDOFF/inputs/BUSBRA_features_ready.csv --output private_outputs/run-001/submissions/gbfrs --jobs 2
```

Place a local copy or symlink to the frozen CSV at
`private_outputs/run-001/inputs/BUSBRA_features_ready.csv`, then:

```bash
uv run --locked python tools/compare_results.py --root private_outputs/run-001
```

Each output path must be new; the CLI refuses overwrites. Real experiments and
comparison status are recorded only in local private outputs. No source data,
patient identifiers, predictions, selected-feature exports or model files are
included in the public repository. The paper release intentionally includes only
aggregate real-data scores and derived plots. Synthetic test scores
are software checks, never research results or CV performance claims.

## Paper and repeat execution — 2026-10-28

All three methods were re-executed sequentially using the unchanged model definitions,
frozen CSV, supplied patient-grouped folds and full 32-candidate search. The private
run manifest records completion of all three methods. The paper release adds
aggregate-only conditional patient-bootstrap intervals and descriptive feature
stability, plus a local LaTeX manuscript and scientific figures. These secondary
analyses were added after the initial results and are not preregistered confirmatory
analyses. See RESULTS.md and STATISTICAL_ANALYSIS.md for limitations.

The original team history and attribution are preserved. Paper author names/order
were supplied by the requester; affiliations, contact details, individual
contributions and research statements require author review before submission.

## Extended evidence — 2026-10-28

Three nested secondary baselines (ANOVA, mutual information and L1 logistic selection)
completed all five supplied BUS-BRA folds. Seven pipelines completed a separate WDBC
cytology benchmark, including an unchanged, hash-pinned author selector called only
from a private local file. No author source is redistributed. The independent
reconstruction and original primary protocol are unchanged. The reference criterion
is now audited directly on synthetic inputs. See SECONDARY_STUDY.md and the validated
aggregate-only secondary results. A different modality benchmark does not establish
external ultrasound validity; author review and submission remain outstanding.

The optional unchanged private author selector also completed BUS-BRA (five nested
outer folds). All seven pipelines are now available for each dataset, with the
original three BUS-BRA pipelines retained as the primary comparison. No reference
run remains pending or is silently replaced.
