# Reproducible method definitions

All methods use only the frozen 288 named predictors, supplied five patient-grouped
outer folds, inner StratifiedGroupKFold(3, shuffle=True, random_state=42+fold),
and the 32 candidates in evaluation_contract.json. StandardScaler is fitted on
each training partition. Fixed selector settings below were chosen before real
data evaluation; no outer scores choose settings. Predictions use SVC's default
zero decision threshold. No resampling, feature engineering, or global filtering.

Ties use mean inner macro-F1, then mean inner accuracy, then mean selected count,
then ParameterGrid order. A scaler/selector is fitted once per inner training
partition and shared only among that partition's SVM candidates. After selection,
a fresh scaler/selector and SVC are fitted to the entire outer training set.
Twenty selector fits per method cover 15 inner training sets and five outer sets.

## All-feature baseline

StandardScaler followed by SVC, retaining all 288 predictors. Linear gamma is
recorded as its actual default `scale` but is ignored by SVC for that kernel.

## Conventional fuzzy-rough QuickReduct

Independent implementation of Jensen and Shen, *New Approaches to Fuzzy-Rough
Feature Selection*, IEEE TFS 17(4), 824–838 (2009),
https://doi.org/10.1109/TFUZZ.2008.924209.
Author manuscript: https://pure.aber.ac.uk/ws/files/3443973/JensenShen.pdf.

Use their similarity Eq. 26: R_a(i,j)=max(1-|x_ia-x_ja|/sigma_a,0), with
training sample standard deviation sigma_a. A constant training feature has
similarity one. Combine with Lukasiewicz T(u,v)=max(u+v-1,0), use implicator
I(u,v)=min(1-u+v,1), and crisp binary decisions. Lower approximation is
inf_j I(R_B(i,j),1[y_j=c]); positive membership is the supremum across classes;
dependency is its mean over training objects (Eqs. 27–28). Reflexivity means only
the object's own class contributes, reducing this exactly to minimum opposite
class dissimilarity. We compute only the cross-class matrix; no training objects
are sampled or discarded. T aggregation is a capped sum of dissimilarities.
The triangular relation satisfies Lukasiewicz transitivity through its metric
construction, without a separate closure changing the relation.

Forward greedy search begins empty and adds the feature with greatest dependency
improvement. Stop on dependency one, exhaustion, or improvement <=1e-12; ties
retain original column order. This is the similarity-relation QuickReduct from
Section III.A. It can return a super-reduct, not a guaranteed minimum subset.
No arbitrary feature-count cutoff or correlation prefilter is applied.

## GBFRS: independent reconstruction of the 2025 preprint

Xia, Lian, Sang, Wang and Gao, *GBFRS: Robust Fuzzy Rough Sets via Granular-ball
Computing*, https://arxiv.org/abs/2501.18413 (v1, 2025).
Journal record: https://doi.org/10.1109/TNNLS.2026.3681370.
Author-linked repository: https://github.com/lianxiaoyu724/GBFRS.
The implemented target is explicitly the accessible preprint, not a claim to
reproduce the journal's numerical tables or the released code's output.

Use fixed purity T=0.90. For ball geometry only, map the standardized training
columns to [0,1] using training minima/ranges; no validation/test observations
enter this calculation. This satisfies the preprint's bounded similarity domain.
The SVM still receives selected StandardScaler columns, not this internal map.

Initial KMeans seeds sample floor(sqrt(n_c)) examples without replacement from
each class c using RandomState(42), following the author archive's refinement of
the preprint's sqrt(n) initialization. KMeans uses explicit seeds and n_init=1.
Ball center is the mean, radius the mean Euclidean distance to that center,
label the majority class (first observed label on ties), purity the majority
fraction (Definition 3). Split impure balls into two using the minimum row-sum
point from each class as initial centers, as in the archive; binary 0/1 encoding
is handled explicitly. Split heterogeneous overlapping balls with two-means
(random_state=5,n_init=1) until center distance >= sum of mean radii. Memberships
are fixed thereafter throughout feature search. Every training row remains in
exactly one ball. Irreducible impurity/overlap raises a reproducibility error.

For subset B, use center distances divided by sqrt(|B|), so GBR_B(i,j)=
1-||c_i[B]-c_j[B]||/sqrt(|B|). On [0,1] columns this lies in [0,1]. Positive
membership is the nearest opposite-label ball distance divided by sqrt(|B|).
Dependency is sum_i (|GB_i|/n)*positive_i, **the own-ball size weights in
Equation 13**. This is the preprint's |B|-dependent distance parameter and
Algorithm 1 stopping convention, rather than a fixed-C minimum reduct.
Greedily add the maximum dependency feature and stop when normalized dependency
fails to increase by >1e-12, or candidates are exhausted. Ties retain column
order. The changing normalizer permits stopping before all columns are used.

The archive has no redistribution license and disagrees with Eq. 13. Its code
is neither copied into src nor imported by the primary reconstruction. See REFERENCE_AUDIT.md for the exact
reference, differences and reproduction checks. Report this method as
**paper-defined GBFRS with documented implementation adaptations**, never as an
unmodified run of the authors' code. These adaptations mean results are not
an exact replication of the authors' experimental protocol.

## Artifacts and timing

The five official CSV/JSON artifacts match the original handoff headers.
internal_run.json additionally records every candidate's three inner scores,
training sizes, selected indices, dependency histories and source checksums.
selector_seconds includes inner and outer selection. refit_seconds includes
outer selection plus scaling and SVC fit, so these two fields overlap and must
not be added. total_seconds is the non-overlapping fold wall time. Runtime
comparison uses total_seconds; candidate workers share memory with native
BLAS/OpenMP limited to one thread. Package versions and hardware are recorded.

The frozen table, identifiers, row-level predictions, model files and per-fold
selected-feature exports stay local. The paper release intentionally publishes
only aggregate scores, descriptive stability, conditional bootstrap intervals
and derived figures. See STATISTICAL_ANALYSIS.md for their interpretation.

## Secondary private reference execution — 2026-10-28

The optional secondary wrapper verifies the inspected author script SHA-256 before
importing it privately, and calls its unchanged selector with training-only geometry,
purity 0.90 and NumPy seed 42. The source is never copied into this repository or
exports. The independent primary reconstruction is unchanged. See SECONDARY_STUDY.md
for completed nested WDBC results, resource-limit reporting, and the synthetic
criterion audit. This is not a reproduction of the original classifier or tables.
