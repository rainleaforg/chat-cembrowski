---
title: Security and Privacy
kind: site
site_path: /docs/security-and-privacy
---

**For technical reviewers**: hospital IT departments, information security teams and privacy
offices evaluating BAPa for institutional use.

Every claim in this document is drawn from the application's source code, and can be checked
against the running application without assistance from the BAPa team.

## Summary

BAPa has two distinct parts, and they have different privacy properties. Conflating them is the
single most common misreading, so this document separates them throughout.

| | **The analysis tool** (`/tool`) | **The research chat** (`/research`) |
|---|---|---|
| Who may use it | Approved accounts only | Anyone, no sign-in |
| What it handles | Your laboratory data file | Typed questions about published research |
| Where processing happens | Entirely inside your browser | On our server and third-party AI services |
| Does data leave the device | **No** | **Yes.** Questions, and dictated audio if used |

**Patient data is only ever loaded into the analysis tool, and the analysis tool does not transmit
it.** This claim is scoped deliberately: it is a statement about `/tool`, not about every page on
the site.

## 1. Where uploaded data is processed

**In the browser, on the user's own machine. The file is never uploaded.**

When a user selects a spreadsheet, the browser hands it to a Web Worker, a background thread inside
the same browser tab. That worker runs a complete Python environment compiled to WebAssembly
(Pyodide), including pandas, NumPy, SciPy and matplotlib. The Sequential Difference Deviation
analysis is performed there, and the resulting graph is drawn there.

At no point is the file, any part of its contents, or any statistic derived from it sent to a
server. There is no server-side analysis component. If the machine were disconnected from the
network after the page finished loading, the analysis would still run to completion.

This is verifiable from outside the application: open the browser's developer tools, switch to the
Network tab, and load a file. No request carrying the file appears, because none is made.

## 2. Whether patient data are transmitted over the internet

**No patient data is transmitted from the analysis tool.**

The application's source contains no network call of any kind within the analysis tool's code: no
`fetch`, no `XMLHttpRequest`, no WebSocket, no `sendBeacon`. The file is passed to the Web Worker by
an in-memory transfer, which is a browser-internal operation and not a network operation.

Three kinds of network traffic do occur while the tool is open. None of them carries file content.

**Loading the analysis engine.** On first use, the browser downloads the Pyodide runtime and its
scientific packages (NumPy, SciPy, pandas, matplotlib, scikit-learn, statsmodels and micropip) from
the public jsDelivr CDN. These are ordinary asset downloads, identical for every user, made before
any file is chosen. The `openpyxl` spreadsheet library and the analysis code itself are served from
BAPa's own origin rather than a package index, specifically so that a network which filters PyPI
does not break the tool.

**Signing in.** Authentication is handled by Clerk. This exchanges account credentials and session
tokens only.

**Crash reports.** If the tool encounters an unrecoverable error, an anonymised report is sent so
the fault can be diagnosed. File names, worksheet names, column headers, cell values, row counts and
anything typed by the user are prohibited from it. Error messages are never transmitted as text —
each is matched against a fixed list of known error shapes and only a category label is sent.
Stack traces are reduced to file-and-line positions with function names discarded. Reports are
capped at five per session, limited to 64 KB, written only to BAPa's own server log, never sent to
a third-party error-tracking service, and never stored in a database.

## 3. What information is stored, locally and remotely

### On the user's device

| What | Where | Contents | Lifetime |
|---|---|---|---|
| Sign-in session | Browser cookies, set by Clerk | Session token and authentication state | Managed by Clerk |
| Page-transition flag | `sessionStorage` | A single character, used to sequence an animation | Deleted on read |
| Research chat transcript | `localStorage` | Questions and answers from `/research` only. Never anything from the analysis tool | Until the user clears it |

**Nothing from the analysis tool is written to any of these.** The application uses no IndexedDB,
no Cache Storage and no service worker. An uploaded file exists only in memory.

### On the servers

The only database is a single table recording the **published posters and papers** in the research
library: title, authors, year, publication, file name, size, content hash, storage key and
processing status. Those files are academic publications uploaded by an administrator. They contain
no patient data, and no user of the analysis tool contributes to that table.

Specifically, the server stores **no** user accounts, **no** email addresses, **no** account
identifiers, **no** IP addresses, **no** chat transcripts and **no** record of tool usage. User
accounts exist in Clerk; the application does not mirror them. Research chat conversations are held
in the browser and re-sent with each question — they are not written to any database.

## 4. How uploaded files are handled during and after analysis

**During.** Files up to 50 MB are read into memory as a byte array and transferred to the Web
Worker. Larger `.xlsx` files are handled differently: a reference to the file is passed to the
worker, which reads ranges from disk on demand rather than loading the whole workbook. Either way
the data reaches Python inside the browser and no further. Results return as a graph image encoded
directly into the page; "Save Analysis" invokes the browser's own print dialogue, so the PDF is
produced on the user's machine with no server-side rendering and no upload.

**After.** Nothing is written to disk and nothing is persisted, so there is nothing to erase.
Closing the tab destroys the worker and everything it held. Two details worth knowing precisely:
"Restart" resets the interface but the worker still holds the parsed file in memory until a
different file is loaded or the tab is closed; "Start the tool over", offered after a crash, does
terminate the worker and discard the file.

## 5. External services

Services contacted while the analysis tool is in use: jsDelivr (delivers the Pyodide runtime and
packages — receives an ordinary asset request, no file data), Clerk (authentication — receives
sign-in credentials and session tokens), and Vercel (hosts the application).

Services contacted by other parts of the site, for completeness: OpenRouter/Google Gemini and
Voyage AI and Qdrant Cloud (research chat — receive the typed question and retrieved passages from
the published corpus), NIH/NLM MedlinePlus and PubMed (research chat, general medical questions —
receive the typed question as a search term), Deepgram (voice input — receives the recorded audio),
Neon/AWS S3/AWS Lambda/OpenAI (publication library and its ingestion — receive published papers and
posters uploaded by an administrator), and Render (hosts the backend service).

**None of the services used by the research chat or publication library are reachable from the
analysis tool.** They belong to different pages, and the analysis tool makes no network calls at
all.

## 6. Browser and platform requirements

A modern browser with WebAssembly support (current Chrome, Edge, Firefox, Safari) and Web Workers
permitted. Outbound HTTPS access to `cdn.jsdelivr.net` is required on first load — this is the
requirement most likely to matter on a restricted hospital network. Roughly a minute for the first
load on a given machine; subsequent loads are served from browser cache. No plugin, extension,
installation or administrator privilege is required — nothing is installed on the workstation.

## 7. Known limitations of the security model

- The analysis engine is fetched from a third-party CDN (jsDelivr) on every cold start; no file
  data is exposed, but it is a runtime dependency on an external provider.
- The application sets no Content-Security-Policy or similar hardening headers; TLS is terminated
  and enforced by the hosting platforms.
- Client-side source maps are published — the original TypeScript is readable by anyone.
- There is no audit log of tool usage beyond Clerk recording sign-ins.
- A user who has been granted access has no way to sign out except by closing the browser or
  clearing cookies.
- Restarting the tool leaves the previous file in browser memory until another file is loaded or
  the tab is closed.
- Access is granted by hand by an administrator, with no automated approval workflow.
- The research chat is deliberately open to anyone, with no sign-in. It should not be used for
  anything patient-identifying, and its voice feature sends audio to a third-party transcription
  service.

## 8. Technical architecture

**The analysis tool** runs a browser tab's Web Worker containing Pyodide (Python compiled to
WebAssembly) with pandas, NumPy, SciPy, matplotlib, scikit-learn and statsmodels, plus the SDD
analysis implementation. The user's file passes from the file input to the Web Worker by an
in-memory transfer and is parsed inside Pyodide; results return as a base64-encoded image. There is
no server in this path.

**The wider application** is a Next.js front end on Vercel, with a separate FastAPI service on
Render answering the research chat and serving the publication library. That service holds a
PostgreSQL table of publication metadata (Neon), a vector index of the published corpus (Qdrant
Cloud), and the PDFs themselves (AWS S3, delivered by short-lived pre-signed links). An AWS Lambda
function ingests newly uploaded publications. **The analysis tool touches none of this.**

**Authentication** is handled by Clerk, with short-lived session tokens independently verified by
the backend service, which holds no Clerk secret and can only verify, never mint. Revoking an
account's access takes effect everywhere within about a minute.

None of this infrastructure ever receives laboratory data, because the analysis tool sends none.

## Questions about this document

Technical or privacy questions, and requests for access to the analysis application, go to
**Dr. George Cembrowski, cembr001@gmail.com**. If assessing BAPa on behalf of an institution, the
companion Access Management document (`/docs/access-management`) sets out how individual access is
requested and granted.
