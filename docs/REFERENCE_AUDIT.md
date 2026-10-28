# Reference verification

Verified on 2026-10-19 against the PDF and author-linked GitHub archive.

## GBFRS

- Preprint v1: https://arxiv.org/pdf/2501.18413 (12 pages).
- Author link on page 1: https://github.com/lianxiaoyu724/GBFRS.
- Inspected author commit: 0382e0899b0dc5529c89d1b9a11da198464d49ad.
- Archive GBFRS_github(2).rar contains only GBFRS_github/GBFRS.py.
- No LICENSE or permission notice is present in the repository or archive.
  The archive stays in ignored private_handoff; it is not vendored or imported by the primary reconstruction.
- Definition 3: mean center, mean-distance radius, majority label/purity.
- Algorithm 1: initial clustering, purity splitting, heterogeneous overlap
  removal, then forward search on fixed ball memberships.
- Equation 13 weights each ball's lower approximation by **its own size/n**.
  The released attribute_reduce instead combines classwise maxima and
  nearest-neighbor-ball weights. It is not algebraically Eq. 13.
- Preprint Algorithm 1 has inconsistent loop/inequality symbols. The accompanying
  prose and released forward search clarify maximum dependency and stopping at
  no increase, using normalization sqrt(number of selected attributes).
- Author source seeds floor(sqrt(class_count)) points per class; purity splits
  choose the lowest feature-sum example in each class. It treats label 1 as
  special, so our binary labels are handled explicitly with no target recoding.
- Our independent reconstruction uses Eq. 13 and correct partition ownership.
  Global overlap rechecks replace the archive's local component queue, preventing
  newly formed overlaps outside a previous component from being skipped. Empty
  clusters are excluded; unsplittable geometry raises an error. These choices,
  deterministic seeds, training-only range normalization and T=0.90 are declared
  adaptations, not claims of source-code numerical equivalence.
- Unit checks verify own-ball weighting on hand-computable unequal-size balls,
  normalized dependency values, disjoint sample coverage, purity, overlap
  removal, and deterministic selected indices. No published benchmark accuracy
  is claimed because source data and a matching licensed benchmark are absent.

## Conventional fuzzy-rough

Jensen and Shen author manuscript, Section III.A, similarity Eq. 26,
Lukasiewicz operators, positive region/dependency Eqs. 27–28 and greedy search.
The six-object manuscript example is a unit fixture: first feature c, then b,
with dependency one after both. Sample standard deviation and a 1e-12
floating-point improvement tolerance are explicit numerical conventions.

## Handoff and data

The supplied four-page PDF and machine contract were inspected. The input CSV's
byte hash, dimensions, labels, patient grouping, finite numerical inputs and
feature allow-list are checked before training. The supplied comparison tool
is preserved separately as an unchanged local acceptance check. It does not
prove nested fitting or validate all timing and parameter details; tests audit
training boundaries in addition to running that checker. The BUS-BRA source
license is absent from this handoff. No source or derived data is redistributed.

## Secondary private reference execution — 2026-10-28

The optional secondary wrapper verifies the inspected author script SHA-256 before
importing it privately, and calls its unchanged selector with training-only geometry,
purity 0.90 and NumPy seed 42. The source is never copied into this repository or
exports. The independent primary reconstruction is unchanged. See SECONDARY_STUDY.md
for completed nested WDBC results, resource-limit reporting, and the synthetic
criterion audit. This is not a reproduction of the original classifier or tables.
