# arXiv source preparation

The manuscript uses a compact venue-neutral title, one centered author line and
standard LaTeX packages. No affiliations or email addresses are invented. The
requester will enter confirmed affiliations/contact details during submission.
The abstract is 1,395 ASCII characters, below arXiv's 1,920-character metadata limit.

Upload a source ZIP with exactly these files at its root/relative paths:

- `main.tex`
- `generated.tex`
- `secondary_generated.tex`
- `expanded_generated.tex`
- `figures/performance-reduction.png`
- `figures/paired-effects.png`
- `figures/label-noise.png`

The bibliography is embedded in `main.tex`; no external `.bib` or `.bbl` is needed.
Use `main.tex` as the top-level document and PDFLaTeX as the processor. The source
contains no engine-specific font setup, shell escape or remote data dependency.
It is compiled locally with Tectonic, including a clean extraction of the ZIP.
The actual arXiv server-generated preview still must be checked during upload.
Do not include local PDFs, build files, instructions, patient data or model files
inside the source ZIP. Keep the PDF separately for inspection.

Suggested primary category: **cs.LG (Machine Learning)**, appropriate here for
feature-selection methodology and applied ML evaluation. **cs.CV** is a possible
cross-list for the imaging application. This is a recommendation, not a promised
classification or acceptance; moderators decide. The system may request an
endorsement. No submission or license selection has been made by this workflow.

Before finalizing, confirm that all named authors approve submission and choose a
license you can grant. Leave journal reference and publication DOI blank unless
there is an actual publication to reference. Review the significant-tool disclosure requirements in the linked moderation
policy before upload. Scientific limitations have not been removed for presentation.

Official guidance checked on 28 October 2026:

- [Category taxonomy](https://arxiv.org/category_taxonomy)
- [Submission overview](https://info.arxiv.org/help/submit/index.html)
- [TeX source requirements](https://info.arxiv.org/help/submit_tex.html)
- [Metadata requirements](https://info.arxiv.org/help/prep.html)
- [Endorsement](https://info.arxiv.org/help/endorsement.html)
- [Moderation policy](https://info.arxiv.org/help/moderation/index.html)
