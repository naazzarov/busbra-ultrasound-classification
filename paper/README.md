# Manuscript draft

**Patient-Grouped Evaluation of Fuzzy-Rough Feature Selection for Breast Ultrasound Classification**

Authors (provided order): Bahodir Nazarov, Zineb Gharib, Fatima Zazoul.
Acknowledgment: Luo Chuan.

This is a research draft for author review, not an accepted or submitted paper.
The study compares three primary pipelines, adds three secondary baselines and a
separate seven-pipeline cytology benchmark, with a
qualified GBFRS reconstruction. It does not claim a new feature-selection
algorithm, clinical validation, equivalence or statistically established superiority.

## Files and provenance

- `main.tex`: complete manuscript; portable LaTeX with embedded bibliography.
- `generated.tex`: primary numeric tables/macros generated from aggregate JSON.
- `secondary_generated.tex`: tables from validated secondary aggregate JSON.
- `../tools/secondary_paper_assets.py`: secondary-table generator.
- `figures/`: scientific plots generated from that same aggregate artifact.
- `../results/aggregate_results.json`: aggregate-only results from the fresh full run.
- `../tools/paper_assets.py`: figure/table generator; optional Matplotlib 3.10.8.

The raw table, patient identifiers, row predictions and private selector/model
artifacts are intentionally absent. The exact prepared CSV must be obtained through
the authorized team channel; the original image archive is not the prepared table.

## Build

Open this directory in Codex's built-in LaTeX editor, or build with an existing
LaTeX installation from this directory. All referenced figures and numeric macros
are included; no bibliography service or external data download is required.

```bash
pdflatex main.tex
pdflatex main.tex
```

Regenerate figures/tables from the repository root using a separate optional figure
environment (the locked model environment is unchanged):

```bash
python -m pip install matplotlib==3.10.8
python tools/paper_assets.py --summary results/aggregate_results.json --output paper
python tools/secondary_paper_assets.py
```

## Author review before submission

The names/order and Luo Chuan acknowledgment came from the requester. Confirm name
spellings, affiliations, contact details, research contributions, funding and
competing-interest statements with all named authors. The authors must review the manuscript and accept responsibility for its contents. No institutional approval or funding has been invented.

The primary and additional-baseline experiments are complete, and the unchanged
private author selector has been evaluated on both BUS-BRA and WDBC. The independent BrEaST cohort now supplies additional method evidence;
frozen-model external testing and human scientific review remain necessary. All conditional intervals
and methodological qualifications should remain visible. `cs.LG` is a plausible
arXiv category; classification, endorsement and acceptance remain arXiv decisions.
No arXiv submission has been made by this workflow.

## Conference literature update

The related-work section includes verified conference publications from AISTATS
2024, ICML 2024, NeurIPS 2025 and MICCAI 2025. These are contextual references, not
methods evaluated in this study. Foundational method citations are retained.
The abstract links directly to the public project repository. See
`../docs/CONFERENCE_REFERENCES.md` for the official publication records.

## Controlled ablation, robustness and BrEaST extension

The draft now includes eleven completed robustness/sensitivity/criterion runs and
six completed pipelines on 252 BrEaST ultrasound patients, using a separately
specified 102-feature extraction. This is independent-cohort method evaluation,
not frozen-model transfer. See `../docs/EXPANDED_STUDY.md`.

`expanded_generated.tex` and `figures/label-noise.png` are additional compilation
inputs. Regenerate them with Matplotlib 3.10.8 using
`python tools/expanded_paper_assets.py` from the repository root.

## arXiv packaging and title layout

The title and author row now use a compact, venue-neutral layout. The source and
PDF metadata use the same author order. The shortened abstract is also suitable
for the arXiv metadata field. See [arXiv source instructions](ARXIV_SUBMISSION.md)
for the exact seven-file bundle, compiler choice and current official guidance.
The upload package is a local deliverable; no submission or acceptance is claimed.
