# Audit of the original GDP prediction notebook

The notebook has 184 cells and contains strong foundational work: a multi-source merge, grouped feature blocks, PCA, repeated VIF diagnostics, panel FE/RE estimation, and separate ML exercises.

## Preserve and refactor

- The original merge logic and variable knowledge.
- Economically coherent PCA blocks: capital, labor, technology, and demographics.
- VIF calculations as diagnostics, with the sequence and final choices explained.
- Fixed- and random-effects comparisons, followed by a formal specification decision.
- ML as a separate predictive extension with time-aware validation.

## Replace in the econometric pipeline

- GDP target reconstruction from `GDP_n` and `real_gdp_ppp_output` (original cell 39): the new main outcome uses observed GDP only.
- Unflagged country-level interpolation of capital components (cell 39): no automatic interpolation occurs.
- Filling `kppp_rppp` missing values with zero (cell 46): zero is retained only when observed; missing remains missing.
- Removing countries or columns by universal missing/zero thresholds (cells 43â€“48): the new pipeline reports coverage; analysis-specific sample selection is documented rather than silently imposed.
- Filling missing inputs with zero for PCA (cell 53): PCA will be fit only on a documented complete-case estimation sample or a separately justified measurement sample.

This preserves the projectâ€™s substantive work while making every treatment and analysis sample auditable.
