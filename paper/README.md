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
competing-interest statements with all named authors. The manuscript explicitly
discloses AI assistance; the authors must review its contents and
accept responsibility. No institutional approval or funding has been invented.

The primary and additional-baseline experiments are complete, and the unchanged
private author selector has been evaluated on both BUS-BRA and WDBC. Independent external ultrasound
validation and human scientific review remain necessary. All conditional intervals
and methodological qualifications should remain visible. `cs.LG` is a plausible
arXiv category; classification, endorsement and acceptance remain arXiv decisions.
No arXiv submission has been made by this workflow.

## Conference literature update

The related-work section includes verified conference publications from AISTATS
2024, ICML 2024, NeurIPS 2025 and MICCAI 2025. These are contextual references, not
methods evaluated in this study. Foundational method citations are retained.
The abstract links directly to the public project repository. See
`../docs/CONFERENCE_REFERENCES.md` for the official publication records.
