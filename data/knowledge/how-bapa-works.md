---
title: How BAPa Works
kind: site
site_path: /tool
---

BAPa's analysis tool performs local, privacy-preserving Sequential Difference Deviation (SDD)
analysis on clinical laboratory result data — currently for troponin and electrolytes, with
Hemoglobin A1c on the roadmap.

## Where the analysis runs

Entirely inside the user's browser. When a lab uploads a spreadsheet (CSV or Excel), the browser
hands it to a Web Worker — a background thread in the same browser tab — running a full Python
environment compiled to WebAssembly (Pyodide), including pandas, NumPy, SciPy and matplotlib. The
SDD analysis is computed there, and the resulting graph is drawn there. The file is never uploaded
to a server: there is no server-side analysis component, and if the network were disconnected after
the page loaded, the analysis would still complete.

## What the method does

SDD analysis works from a laboratory's own paired patient results over time, rather than external
control material, to quantify preanalytic and analytic variation and flag when a testing process
has drifted. This is patient-based quality control: the statistical method behind it is
Dr. George Cembrowski's research, developed with Jenna Rimkus, and BAPa is what makes it directly
usable — a lab runs the analysis on its own data without needing to implement the statistics
itself.

## The two parts of the site

The analysis tool (`/tool`) and the research chat (`/research`) are different parts of BAPa with
different privacy properties: the tool processes uploaded laboratory data locally and transmits
nothing; the chat is an open research assistant that sends typed questions to a server. See the
security and privacy documentation for the full detail.

## What's not built yet

The `/research/background`, `/methodology`, `/results`, and `/citations` pages are placeholders —
they don't have content yet. Don't tell a user those pages already explain something; point them to
`/presentation` for published research and `/docs/security-and-privacy` for architecture detail
instead.
