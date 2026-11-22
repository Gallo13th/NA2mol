"""Split the journal-agnostic manuscript package into the Nature Biotechnology file set.

NBT wants three separate files: the manuscript (main text + Methods + main figures/tables), an
Extended Data PDF (at most ten display items), and one combined Supplementary Information PDF.
The single PDF we maintained until now merges all three, which is fine for a working draft but not
submittable.

This script does the mechanical part once, reproducibly:

  * renumbers every Supplementary item to its Extended Data or Supplementary slot;
  * replaces \\ref{} to main-paper labels with literal numbers, because the SI no longer compiles
    together with the manuscript;
  * pulls the model-configuration table out of the notes file into its own Extended Data file;
  * writes the two standalone masters and strips the SI block from the manuscript.

Run once after the allocation is settled; the output files are what the build script then compiles.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# old Supplementary table number -> Extended Data table number
ED_TABLES = {1: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 8, 14: 9}
# old Supplementary table number -> new Supplementary table number
SUP_TABLES = {2: 1, 3: 2, 4: 3, 5: 4, 13: 5}

# main-paper labels referenced from the SI: literal numbers in the current manuscript
MAIN_REFS = {
    'radar_values_tab': '1',      # Table 1
    'overlap_table': '2',         # Table 2
    'benchmark_pair_fig': '2',    # Fig. 2
    'ablation_fig': '3',          # Fig. 3
    'kras_case_fig': '4',         # Fig. 4
    'alg:constrained': '3',       # Algorithm 3
    'eq:recovery_curve': '9',     # Eq. (9)
    'eq:recovery': '10',          # Eq. (10)
}

SI_FILES = sorted(str(p.name) for p in ROOT.glob('si_*.tex'))


def renumber_tables(text: str) -> str:
    """Renumber headings and inline references in one pass, without re-mapping our own output.

    Headings are swapped for tokens first: otherwise a heading rewritten to
    "Supplementary Table 1" would be picked up again by the inline rule and mapped a second time.
    """
    tokens: list[tuple[str, str]] = []

    def head(m: re.Match) -> str:
        n = int(m.group(1))
        if n in ED_TABLES:
            final = f'\\section*{{Extended Data Table {ED_TABLES[n]} | {m.group(2)}}}'
        else:
            final = f'\\section*{{Supplementary Table {SUP_TABLES[n]} | {m.group(2)}}}'
        tokens.append((f'@@H{len(tokens)}@@', final))
        return f'@@H{len(tokens) - 1}@@'

    def fig_head(m: re.Match) -> str:
        tokens.append((f'@@H{len(tokens)}@@', f'\\section*{{Extended Data Fig. 1 | {m.group(1)}}}'))
        return f'@@H{len(tokens) - 1}@@'

    text = re.sub(r'\\section\*\{Supplementary Table (\d+) \| (.*?)\}', head, text)
    text = re.sub(r'\\section\*\{Supplementary Fig\. (\d+) \| (.*?)\}', fig_head, text)

    def inline(m: re.Match) -> str:
        n = int(m.group(1))
        if n in ED_TABLES:
            return f'Extended Data Table~{ED_TABLES[n]}'
        return f'Supplementary Table~{SUP_TABLES[n]}'

    text = re.sub(r'Supplementary Table[~ ](\d+)', inline, text)
    text = re.sub(r'Supplementary Fig\.[~ ]1',
                  lambda m: 'Extended Data Fig.~1', text)

    for token, final in tokens:
        text = text.replace(token, final)
    return text


def literals(text: str) -> str:
    for label, num in MAIN_REFS.items():
        text = text.replace(f'\\ref{{{label}}}', num)
    return text


def main() -> int:
    # 1. renumber + de-reference every SI item file
    for name in SI_FILES:
        p = ROOT / name
        s = p.read_text()
        p.write_text(literals(renumber_tables(s)))

    # 2. split the notes file: notes -> supplementary, Table 1 -> extended data
    notes = ROOT / 'supplementary_notes_revised.tex'
    text = notes.read_text()
    start = text.index('\\section*{Supplementary Tables}')
    head, tail = text[:start], text[start:]
    # drop the table heading and the \input lines, keep the table body
    body = re.sub(r'\\section\*\{Supplementary Tables\}\s*', '', tail, flags=re.S)
    body = re.sub(r'\\input\{[^}]*\}\s*', '', body)
    table_start = body.index('\\section*{Supplementary Table 1 |')
    table_txt, notes_tail = body[table_start:], body[:table_start]
    (ROOT / 'ed_table1.tex').write_text(
        table_txt.replace('\\section*{Supplementary Table 1 |',
                          '\\section*{Extended Data Table 1 |', 1).strip() + '\n')
    notes.write_text(literals((head + notes_tail).rstrip()) + '\n')
    # the algorithm counter lives inside the SI, so \ref{alg:constrained} must stay a \ref
    print('split supplementary_notes_revised.tex -> notes + ed_table1.tex')

    # 3. standalone masters
    preamble = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
\usepackage{url}
\usepackage[colorlinks=false,hidelinks]{hyperref}
\graphicspath{{figures/}}
\renewcommand{\thefigure}{ED~\arabic{figure}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
"""
    ed_items = '\n'.join([
        r'\input{si_fig_mapping_control}',
        r'\clearpage',
        r'\input{ed_table1}',
        r'\input{si_ed_table1}',
        r'\input{si_ed_table2}',
        r'\input{si_ed_table3}',
        r'\input{si_ed_table4}',
        r'\input{si_ed_table5}',
        r'\input{si_ed_table6}',
        r'\input{si_table_radar_values}',
        r'\input{si_table_mapping_control}',
    ])
    title = ('Bridging protein and nucleic-acid pockets via shared physicochemical '
             'representations for 3D molecular generation with NA2smol')
    (ROOT / 'extended_data.tex').write_text(
        preamble
        + '\\title{\\vspace{-1.2em}\\textbf{Extended Data}\\par\\vspace{0.6em}'
          '\\large ' + title + '}\n\\date{}\n\\begin{document}\n\\maketitle\n\n'
        + ed_items + '\n\\end{document}\n')
    print('wrote extended_data.tex (1 figure + 9 tables)')

    sup_items = '\n'.join([
        r'\input{supplementary_notes_revised}',
        r'\clearpage',
        r'\input{si_revision_v2_revised}',
        r'\input{si_table_overlap}',
    ])
    (ROOT / 'supplementary.tex').write_text(
        preamble
        + '\\title{\\vspace{-1.2em}\\textbf{Supplementary Information}\\par\\vspace{0.6em}'
          '\\large ' + title + '}\n\\date{}\n\\begin{document}\n\\maketitle\n\n'
        + sup_items + '\n\\end{document}\n')
    print('wrote supplementary.tex (6 notes + 5 tables)')

    # 4. strip the SI block from the manuscript
    man = ROOT / 'sn-article-revised.tex'
    s = man.read_text()
    s = re.sub(r'\\section\*\{Supplementary Information\}\s*\\input\{supplementary_notes_revised\}\s*', '',
               s)
    s = literals(renumber_tables(s))
    man.write_text(s)
    print('stripped the SI block from sn-article-revised.tex and updated its ED/SI citations')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
