# NA2smol manuscript — Overleaf project

Bridging protein and nucleic-acid pockets via shared physicochemical representations for 3D molecular
generation with NA2smol

## What is here

| File | Role |
|---|---|
| `sn-article-revised.tex` | **main file** — manuscript: main text, Methods, Figures 1–5, Table 1, back matter |
| `extended_data.tex` | Extended Data master (Extended Data Fig. 1 + Extended Data Tables 1–9) |
| `supplementary.tex` | Supplementary Information master (Notes 1–6, Tables 1–6) |
| `sn-jnl.cls`, `bst/sn-mathphys-num.bst` | Springer Nature LaTeX class and bibliography style |
| `sn-bibliography.bib` | references |
| `figures/` | the five main figure PDFs and the Extended Data figure |
| `si_*.tex`, `ed_table1.tex`, `supplementary_notes_revised.tex`, `table_radar_values.tex`, `table_overlap.tex` | section and table files included by the three masters |

## How to compile

Overleaf: set **Main document = `sn-article-revised.tex`**, compiler **pdfLaTeX**, and build once so that
BibTeX runs (Overleaf does this automatically).

Locally:

```
pdflatex sn-article-revised
bibtex   sn-article-revised
pdflatex sn-article-revised
pdflatex sn-article-revised
```

The other two masters are self-contained:

```
pdflatex extended_data      # Extended Data, 10 pages
pdflatex supplementary      # Supplementary Information, 9 pages
```

Both need two passes; neither uses BibTeX.

## Notes for editing

- Numbering is manual in places (`Eqs. (9)–(10)`, `Extended Data Table 6`, `Supplementary Table 6`) because the
  three documents are separate PDFs; if you renumber a table or equation, search for the plain-text reference.
- Figures are included at their final print size; please do not rescale them, the panel text is set for that width.
- Page numbers that the class places inside a float caption are repaired by `scripts/fix_page_numbers.py` in the
  full repository; this does not affect Overleaf compilation of the text itself.
