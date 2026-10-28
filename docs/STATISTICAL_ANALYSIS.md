# Statistical analysis and interpretation

The primary endpoint remains the mean macro-F1 across the five supplied outer
folds. Report its sample SD; do not interpret this SD as a standard error or a
confidence interval. Patient grouping is retained in every evaluation split.

The publication analysis additionally pools the unchanged out-of-fold class
predictions. It resamples the 1,064 patients within diagnosis strata with
replacement, retains all images for each drawn patient, and uses the identical
resample for all three methods. With 5,000 resamples and NumPy seed 42, percentile
2.5/97.5 limits describe each pooled macro-F1 and the paired method differences.
Each difference refers to the **pooled** metric, not the mean of fold metrics.
No model is refitted, tuned or selected in this analysis.

These are **conditional intervals for the fixed OOF predictions**. They do not
account for all uncertainty from retraining the learning procedure or dependence
among overlapping CV training sets. They are not a full nested bootstrap, a test
of clinical superiority, or a non-inferiority analysis. No p-values or equivalence
claims are supplied. A narrow point difference and an interval spanning zero
cannot establish equivalence.

Selection stability uses all ten pairwise Jaccard overlaps of the five outer
feature sets and Nogueira et al.'s chance-adjusted sample-variance measure. The
latter is computed on a five-by-288 inclusion matrix using variance with ddof=1.
These are descriptive summaries of overlapping outer training sets. They are
not independent biological replications. Named features include 42 redundant
measurements, so alternative but equivalent columns can reduce named-set overlap.

The main rerun executes methods sequentially with the same jobs setting and
native thread limit. Wall times remain machine/load dependent; ordinary desktop
processes are not controlled. The earlier concurrent run is retained as a prior
software-reproduction record, not used as an isolated speed benchmark.

`busbra summarize` validates the exact dataset fingerprint, method/protocol/status,
prediction schema, image/group/truth/fold alignment, allowed selected names,
baseline feature count and positive fold times before producing aggregate JSON.
This is the only experiment artifact intentionally published with the paper.
The frozen table, patient identifiers, row-level predictions, per-fold selected
feature exports and complete private audit records remain excluded from GitHub.
