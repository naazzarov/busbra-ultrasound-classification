# Manuscript draft

**Patient-Grouped Evaluation of Fuzzy-Rough Feature Selection for Breast Ultrasound Classification**

Authors (provided order): Bahodir Nazarov, Zineb Gharib, Fatima Zazoul.
Acknowledgment: Luo Chuan.

This is a research draft for author review, not an accepted or submitted paper.
The study is an empirical comparison of three existing method families, with a
qualified GBFRS reconstruction. It does not claim a new feature-selection
algorithm, clinical validation, equivalence or statistically established superiority.

## Files and provenance

- `main.tex`: complete manuscript; portable LaTeX with embedded bibliography.
- `generated.tex`: exact numeric tables/macros generated from the aggregate JSON.
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
```

## Author review before submission

The names/order and Luo Chuan acknowledgment came from the requester. Confirm name
spellings, affiliations, contact details, research contributions, funding and
competing-interest statements with all named authors. The manuscript explicitly
discloses substantial Codex assistance; the authors must review its contents and
accept responsibility. No institutional approval or funding has been invented.

The code/experiment are complete, but additional external validation and reference
implementation checks would strengthen the research. All conditional intervals
and methodological qualifications should remain visible. `cs.LG` is a plausible
arXiv category; classification, endorsement and acceptance remain arXiv decisions.
No arXiv submission has been made by this workflow.
