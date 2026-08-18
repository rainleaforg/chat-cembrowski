---
title: Using the Analysis Tool
kind: site
site_path: /tool
---

The analysis tool requires signing in (Google account) and administrator-approved access. Once
approved, go to the site and press **Get Started!**, or open the **App** tab. The first load takes
around a minute while the analysis engine downloads; later visits are faster.

The tool walks through six steps. Every step has a **Help** button with detail specific to that
step, and an **Overview** button explaining the whole sequence.

1. **Load Dataset** — Upload a `.csv` or Excel file with one row per test result and clear column
   headers. If an Excel workbook has more than one tab, choose which worksheet to analyze. All
   analysis runs locally in the browser.
2. **Inspect Data** — Look at the first and last rows of the file to see what each column actually
   contains, before mapping columns — useful because exports often have generic headers. This step
   is optional.
3. **Column Mapping** — Select which uploaded columns correspond to Patient ID, Date/Time, and
   Result (all required), plus optional columns for Age, Sex/Gender, and Inpatient/Outpatient.
4. **Dataset Checks** — Scans the mapped data for missing values, duplicate rows, and unusual
   result entries (e.g. non-numeric values where a result was expected), with warnings and
   recommendations before proceeding.
5. **Analysis Settings** — Choose an analysis profile — **Troponin** or **Electrolytes** are
   available today; **Hemoglobin A1c** is on the roadmap and not yet selectable. Set thresholds,
   pairing limits, and result-handling and display options; defaults work if no changes are needed.
6. **Results** — The SDD analysis graph. **Save Analysis** produces a PDF via the browser's own
   print dialogue, generated on the user's machine with no server involved. **Restart** starts over.

Patient data never leaves the browser at any step — see the security and privacy documentation for
the full technical detail on how that's enforced.
