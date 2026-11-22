"""Regenerate presubmission_supplement.tex with the captions currently in the manuscript.

The supplement reproduces the main figures and tables for a presubmission enquiry, so its
captions must be the manuscript's captions verbatim. Copying them by hand is what let the
supplement drift out of step before, so they are extracted here on every build: the figure and
table bodies are pulled from their source files and any cross-reference is resolved to a literal
number, because the supplement has its own numbering.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'sn-article-revised.tex'
TABLE1 = ROOT / 'table_radar_values.tex'


def brace_span(text: str, open_idx: int) -> str:
    depth, out, k = 1, [], open_idx + 1
    while depth:
        c = text[k]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        if depth:
            out.append(c)
        k += 1
    return ''.join(out)


def caption_before(text: str, label: str) -> str:
    i = text.index('\\label{' + label + '}')
    j = text.rindex('\\caption{', 0, i)
    return brace_span(text, j + len('\\caption'))


FIG = {'overview': '1', 'benchmark_pair_fig': '2', 'ablation_fig': '3',
       'clivia_case_fig': '4', 'kras_case_fig': '5'}
TAB = {'radar_values_tab': '1'}


def resolve_refs(cap: str) -> str:
    def sub(m):
        key = m.group(1)
        if key in TAB:
            return 'Table~' + TAB[key]
        if key in FIG:
            return 'Fig.~' + FIG[key]
        return {'sec2': 'Results', 'sec3': 'Discussion'}.get(key, key)
    return re.sub(r'\\ref\{([^}]+)\}', sub, cap)



def drop_si_pointers(cap: str) -> str:
    """Remove references to supplementary material from a caption.

    The presubmission supplement carries only the main figures and tables, so a caption that
    says "breakdown in Supplementary Table 5" points at something the reader does not have.
    Only the pointer itself is cut: the sentence keeps whatever it said before the pointer, so
    a substantive fact in the same sentence is not lost with it.
    """
    # resolving \ref inside "Table~\ref{...}" produced a doubled prefix
    cap = re.sub(r'Table~Table~', 'Table~', cap)
    cap = re.sub(r'Figs?\.~Figs?\.~', 'Fig.~', cap)

    # a pointer introduced by a semicolon inside a parenthetical: drop the clause, keep the group
    cap = re.sub(r';\s*[^();]*?(?:Supplementary|Extended Data) (?:Table|Fig|Note)[^();]*?(?=\))', '', cap)
    # a pointer appended with "and ..." to a sentence that already says something
    cap = re.sub(r',?\s+and\s+(?:the\s+)?[^.;]*?(?:Supplementary|Extended Data) (?:Table|Fig|Note)[^.;]*', '', cap)
    # a semicolon clause that only redirects the reader
    cap = re.sub(r';\s*[^.;]*?(?:Supplementary|Extended Data) (?:Table|Fig|Note)[^.;]*', '', cap)
    # a sentence whose only content is "X are in Supplementary Table N"
    parts = re.split(r'(?<=[.])\s+', cap)
    parts = [x for x in parts
             if not (re.search(r'(?:Supplementary|Extended Data) (?:Table|Fig|Note)', x)
                     and re.search(r'\b(?:are|is)\s+(?:given\s+)?in\s+Supplementary', x))]
    cap = ' '.join(parts)
    # a pointer in its own short sentence, or a parenthetical that is nothing but a pointer
    cap = re.sub(r'\([^()]*?(?:Supplementary|Extended Data) (?:Table|Fig|Note)[^()]*?\)', '', cap)
    cap = re.sub(r'\s*(?:See|Given in|Full (?:caption|detail)s? in)[^.;]*?(?:Supplementary|Extended Data) (?:Table|Fig|Note)[^.;]*\.', '', cap)

    # once the SI pointer is gone, a trailing sentence that points at the same main table as
    # the sentence before it says nothing new
    cap = re.sub(r'(given in Table~(\d+)\.)\s+The raw values behind each axis are given in '
                 r'Table~\2\.', r'\1', cap)

    cap = re.sub(r'\s+', ' ', cap)
    cap = re.sub(r'\s+([.,;:])', r'\1', cap)
    cap = re.sub(r'\(\s*;\s*', '(', cap)
    cap = re.sub(r';\s*\)', ')', cap)
    cap = re.sub(r'\(\s*\)', '', cap)
    return cap.strip()


def _sentences(s: str) -> list[str]:
    """Split on sentence-final periods while protecting decimal points."""
    protected = re.sub(r'(?<=\d)\.(?=\d)', '\x00', s)
    return [p.replace('\x00', '.') for p in re.split(r'(?<=[.])\s+', protected)]


def clean_table_file(src, dest):
    """Write a copy of a table file whose caption carries no supplementary pointers.

    The main-text table files are correct as they stand: their cross-references point at the
    submitted supplementary material. The presubmission excerpt has none of that material, so
    it needs the same table with the pointers removed.
    """
    text = src.read_text()
    tag = chr(92) + 'caption{'
    i = text.index(tag)
    depth, k = 1, i + len(tag)
    while depth:
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
        k += 1
    caption = text[i + len(tag):k - 1]
    # Drop whole sentences that only point at supplementary material; clause-level surgery
    # left dangling fragments such as "Success." when the pointer shared a sentence.
    caption = ' '.join(p for p in _sentences(caption) if not re.search(r'Supplementary|Extended Data', p))
    text = text[:i] + tag + drop_si_pointers(caption) + '}' + text[k:]
    # The table note below the tabular follows the same rule.
    head, sep, tail = text.partition('\\end{tabular}')
    if sep and re.search(r'Supplementary|Extended Data', tail):
        tail = ' '.join(p for p in _sentences(tail) if not re.search(r'Supplementary|Extended Data', p))
        text = head + sep + tail
    dest.write_text(text)


def main() -> int:
    main_tex = SRC.read_text()
    table1_tex = TABLE1.read_text()

    caps = {n: drop_si_pointers(resolve_refs(caption_before(main_tex, k)))
            for k, n in FIG.items()}
    caps['table1'] = drop_si_pointers(
        resolve_refs(caption_before(table1_tex, 'radar_values_tab')))

    doc = r"""% Exhibits accompanying the presubmission enquiry: the five main figures and the main
% table, with the full captions as they appear in the manuscript.
% Generated by build_supplement_tex.py - edit that script, not this file.
\documentclass[11pt]{article}
\usepackage[a4paper,top=1.6cm,bottom=1.6cm,left=1.5cm,right=1.5cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{amsmath,amssymb}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{hyperref}
\hypersetup{hidelinks}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em}
\pagestyle{empty}

% the sn-mathphys-num style expects helpers provided by the sn-jnl class
\providecommand{\citeauthoryear}[2]{#1, #2}
\providecommand{\citenum}[1]{#1}
\providecommand{\bibcommenthead}{}

\newcommand{\exhibit}[1]{%
  \par\vspace{0.5em}
  {\footnotesize \textbf{Figure.}~#1}\par
}

\begin{document}

{\footnotesize Supplement to the presubmission enquiry \hfill 22 November 2026}\\[-0.25em]
\rule{\textwidth}{0.4pt}
\vspace{0.5em}

{\bfseries\large Bridging protein and nucleic-acid pockets via shared physicochemical
representations for 3D molecular generation with NA2smol --- main figures and table}

\vspace{0.3em}
{\footnotesize Letian Gao, Yi Li, Wenbiao Zhou, Jiahao Chen, Bing Su, Xing Li and Zhi John Lu}\\[-0.15em]
{\footnotesize \emph{Tsinghua University; Renmin University of China; Institute of Zoology, Chinese Academy of Sciences}}\\[-0.15em]
{\footnotesize Equal contribution: Letian Gao, Yi Li, Wenbiao Zhou and Jiahao Chen. Corresponding authors: Bing Su, Xing Li and Zhi John Lu.}

\vspace{0.8em}
{\footnotesize\emph{This supplement reproduces the five main figures and the main table with the
captions as submitted with the manuscript.}}\par
\vspace{0.4em}
{\footnotesize
\textbf{Figure 1} Workflow, training and candidate prioritisation.\par
\textbf{Figure 2} Paired benchmark evaluation on the DUD-E protein benchmark and the RNA test set.\par
\textbf{Figure 3} Comparison of the four NA2smol training configurations.\par
\textbf{Figure 4} Structure-based design of fluorogenic ligands for the Clivia aptamer.\par
\textbf{Figure 5} Experimental validation of a ligand candidate for the KRAS promoter G-quadruplex.\par
\textbf{Table 1} Metric values for both benchmarks and the four training configurations.\par
}

\clearpage
\section*{Figure 1}
\includegraphics[width=0.98\textwidth,height=0.60\textheight,keepaspectratio]{figures/figure1_nbt.pdf}
\exhibit{__FIG1__}

\clearpage
\section*{Figure 2}
\includegraphics[width=0.98\textwidth,height=0.60\textheight,keepaspectratio]{figures/figure2_nbt.pdf}
\exhibit{__FIG2__}

\clearpage
\section*{Figure 3}
\includegraphics[width=0.98\textwidth,height=0.60\textheight,keepaspectratio]{figures/figure3_nbt.pdf}
\exhibit{__FIG3__}

\clearpage
\section*{Figure 4}
\includegraphics[width=0.98\textwidth,height=0.60\textheight,keepaspectratio]{figures/figure5_clivia.pdf}
\exhibit{__FIG4__}

\clearpage
\section*{Figure 5}
\includegraphics[width=0.98\textwidth,height=0.60\textheight,keepaspectratio]{figures/figure5_kras.pdf}
\exhibit{__FIG5__}

\clearpage
\section*{Table 1}
\input{presub_table1}

\clearpage
\bibliographystyle{bst/sn-mathphys-num}
\bibliography{sn-bibliography}

\end{document}
"""
    doc = doc.replace('__FIG1__', caps['1']).replace('__FIG2__', caps['2'])
    doc = doc.replace('__FIG3__', caps['3']).replace('__FIG4__', caps['4'])
    doc = doc.replace('__FIG5__', caps['5'])
    clean_table_file(ROOT / 'table_radar_values.tex', ROOT / 'presub_table1.tex')
    (ROOT / 'presubmission_supplement.tex').write_text(doc)

    # the Word builder reads the same captions from this cache
    import json
    # The Word copy has no reference list, so drop LaTeX citation commands from its captions
    # rather than leaking raw BibTeX keys such as 'bemis1996' into the docx.
    plain = lambda text: re.sub(r'\s+', ' ', re.sub(r'\\cite\{[^}]*\}', '', text)).strip()
    json_caps = {f'fig{n}': plain(caps[n]) for n in '12345'} | {
        'table1': plain(caps['table1'])}
    Path('/tmp/full_caps.json').write_text(json.dumps(json_caps, indent=1))
    print('presubmission_supplement.tex regenerated with current captions')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
