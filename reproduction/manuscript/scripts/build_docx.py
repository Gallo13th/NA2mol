"""Build the Word supplement for the presubmission enquiry.

The enquiry letter itself is now the corresponding author's Word document
(`presubmission_enquiry.docx`); this script rebuilds only the supplement.

The layout follows the compiled PDFs: same paper size and margins, Times-family text, bold
section headings, full-width figures and the complete captions. Kept as a script so the Word
files can be regenerated whenever the manuscript changes, rather than being one-off exports.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
TEXT_WIDTH_CM = 21.0 - 2.1 - 2.1          # A4 width less the enquiry side margins

BODY_PT = 10.5
CAPTION_PT = 7.6
HEAD_PT = 11.5


# ── helpers ────────────────────────────────────────────────────────────────
def detex(x: str) -> str:
    """Turn a LaTeX fragment into plain text for Word."""
    x = re.sub(r'\\textbf\{([^{}]*)\}', r'\1', x)
    x = re.sub(r'\\emph\{([^{}]*)\}', r'\1', x)
    x = re.sub(r'\\textit\{([^{}]*)\}', r'\1', x)
    x = re.sub(r'\\url\{([^{}]*)\}', r'\1', x)
    x = re.sub(r'\\AA\b', 'A', x)
    x = x.replace(r'\textbf', '').replace(r'\emph', '')
    x = x.replace('~', ' ')
    x = x.replace(r'\%', '%').replace(r'\&', '&').replace(r'\_', '_')
    x = x.replace('$\\mu$', '\u00b5').replace(r'\mu', '\u00b5')
    x = x.replace('$', '')
    x = re.sub(r'\\[a-zA-Z]+\{([^{}]*)\}', r'\1', x)
    x = re.sub(r'\\[a-zA-Z]+', '', x)
    x = x.replace('{', '').replace('}', '')
    x = re.sub(r'\s+', ' ', x)
    return x.strip()


def page_setup(doc: Document, top: float, bottom: float, left: float, right: float) -> None:
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.top_margin, s.bottom_margin = Cm(top), Cm(bottom)
    s.left_margin, s.right_margin = Cm(left), Cm(right)
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(BODY_PT)
    style.paragraph_format.space_after = Pt(4)
    style.paragraph_format.space_before = Pt(0)
    # East Asian font fallback so the file opens cleanly everywhere
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts'); rpr.append(rfonts)
    rfonts.set(qn('w:eastAsia'), 'Times New Roman')


def para(doc, text='', size=BODY_PT, bold=False, italic=False, align=None,
         space_before=0, space_after=4, left_indent=None, hanging=None):
    p = doc.add_paragraph()
    if text:
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    if left_indent is not None:
        pf.left_indent = Cm(left_indent)
    if hanging is not None:
        pf.first_line_indent = Cm(-hanging)
    return p


def heading(doc, text, before=8, after=2):
    return para(doc, text, size=HEAD_PT, bold=True, space_before=before, space_after=after)


def rule(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    pPr = p._p.get_or_add_pPr()
    borders = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single'); bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1'); bottom.set(qn('w:color'), '000000')
    borders.append(bottom); pPr.append(borders)


def bullet(doc, text):
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(text)
    run.font.size = Pt(BODY_PT)
    run.font.name = 'Times New Roman'
    pf = p.paragraph_format
    pf.space_before = Pt(0); pf.space_after = Pt(2)
    pf.left_indent = Cm(0.7); pf.first_line_indent = Cm(-0.35)
    return p




def braced(t: str, open_idx: int) -> str:
    """Return the contents of the brace group that starts at open_idx."""
    depth, out, k = 1, [], open_idx + 1
    while depth and k < len(t):
        c = t[k]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        if depth:
            out.append(c)
        k += 1
    return ''.join(out)


def multicolumn_text(row: str) -> str:
    """Third brace group of \\multicolumn{cols}{spec}{text}, as plain text."""
    i = row.index('\\multicolumn')
    j = row.index('{', i + len('\\multicolumn'))
    for _ in range(2):                     # skip the cols and spec arguments
        j = row.index('{', j + len(braced(row, j)) + 2)
    return detex(braced(row, j))



def strip_rules(row: str) -> str:
    """Drop booktabs rules that stick to the front of a row after splitting on \\\\."""
    return re.sub(r'^(?:\\midrule|\\cmidrule(?:\([^)]*\))?\{[^}]*\}|\\addlinespace(?:\[[^\]]*\])?)\s*',
                  '', row).strip()


def set_col_widths(table, widths_cm):
    """Word ignores table autofit unless every cell carries the width."""
    table.autofit = False
    tblPr = table._tbl.tblPr
    layout = OxmlElement('w:tblLayout')
    layout.set(qn('w:type'), 'fixed')
    tblPr.append(layout)
    for row in table.rows:
        for cell, w in zip(row.cells, widths_cm):
            cell.width = Cm(w)
    for col, w in zip(table.columns, widths_cm):
        col.width = Cm(w)


# ── the enquiry ────────────────────────────────────────────────────────────
def build_enquiry() -> Path:
    doc = Document()
    page_setup(doc, 1.6, 1.5, 2.1, 2.1)

    # running header line: label left, date right, then a rule
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(2)
    tab = pf.tab_stops
    tab.add_tab_stop(Cm(TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT)
    r = p.add_run('Presubmission enquiry\t10 November 2026')
    r.font.size = Pt(9)
    rule(doc)

    para(doc, 'Bridging protein and nucleic-acid pockets via shared physicochemical representations for 3D molecular generation with NA2smol',
         size=15, bold=True, space_before=4, space_after=3)
    para(doc, 'Letian Gao, Yi Li, Wenbiao Zhou, Jiahao Chen, Bing Su, Xing Li and Zhi John Lu', size=9, space_after=2)
    para(doc, 'Tsinghua University; Renmin University of China; Institute of Zoology, Chinese Academy of Sciences', size=9, italic=True, space_after=2)
    para(doc, 'Corresponding authors: Bing Su, Xing Li and Zhi John Lu.', size=8, space_after=6)

    heading(doc, 'Summary')
    para(doc, "Structure-based molecular generation is routine for protein pockets but has not "
              "been established for nucleic acids. The obstacle is not only the scarcity of "
              "complex structures; most pocket-conditioned generators encode a pocket through "
              "amino-acid-specific features for which there is no nucleotide analogue, so a "
              "protein-trained model cannot simply be pointed at an RNA site. We report NA2smol, "
              "a Bayesian flow network that conditions on descriptors shared by both target "
              "classes, adds a ligand-only electron-density pretraining stage and a "
              "teacher\u2013student alignment loss, and generates three-dimensional ligands for "
              "either class of binding site.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    heading(doc, 'Principal claims')
    for b in [
        "A pocket representation restricted to descriptors that proteins and nucleic acids share "
        "(pharmacophore type, hybridisation state, van der Waals radius, partial charge) "
        "transfers across the two target classes without residue-specific features.",
        "On an RNA test set NA2smol retained 239 molecules across 17 of 27 evaluable targets "
        "under the most stringent tier of a four-stage screening cascade, 0.68% of 35,358 valid "
        "generations.",
        "Against seven structure-based generators, NA2smol leads retrospective ECFP4 recovery "
        "on both benchmarks: 0.218 against 0.080 for the next method on RNA, and 0.656 against 0.482 on "
        "the protein benchmark. On conformational strain it is mid-pack rather than leading, and "
        "we report the fraction of molecules whose geometry could be evaluated separately from "
        "the fraction passing a strain threshold.",
        "A KRAS promoter G-quadruplex case study carries the output through fragment "
        "recombination and ensemble scoring to a tested compound: a bis-benzothiazole that "
        "reduced normalised thioflavin T fluorescence in a concentration-dependent manner, "
        "apparent half-maximal concentration 22.4 \u00b5M.",
    ]:
        bullet(doc, b)

    heading(doc, 'Why this journal')
    para(doc, "The contribution is a computational design question rather than a single "
              "application: how to build a representation and a training strategy that transfer "
              "between two chemically different target classes when labelled structural data "
              "exist for only one. Nucleic-acid pockets are a demanding test of that question, "
              "and the work is reported end to end, from pretraining to a tested compound, with "
              "the code and derived data released.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    heading(doc, 'Enclosed exhibits')
    para(doc, "The five main figures and the main table, with their full captions and the "
              "reference list, are supplied in the accompanying supplement "
              "(presubmission_supplement.docx) so that the complete evidence can be assessed "
              "alongside this summary.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    heading(doc, 'Status')
    para(doc, "Original work, not published or under consideration elsewhere. All authors have "
              "approved submission; we declare no competing interests. The manuscript is [XX] "
              "pages with five main figures, one main table and [K] supplementary items.",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    para(doc, 'Zhi John Lu', size=9, space_before=10, space_after=0)
    para(doc, 'Associate Professor, School of Life Sciences, Tsinghua University, on behalf of all authors', size=9, space_after=0)

    # Kept only as a reference template; the canonical enquiry is the author-maintained
    # presubmission_enquiry.docx and must not be overwritten by a rebuild.
    out = ROOT / 'presubmission_enquiry_template.docx'
    doc.save(out)
    return out


# ── the supplement ─────────────────────────────────────────────────────────
def build_supplement() -> Path:
    caps = json.loads(Path('/tmp/full_caps.json').read_text())
    doc = Document()
    page_setup(doc, 1.6, 1.6, 1.5, 1.5)
    width = Cm(21.0 - 3.0)

    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(2)
    pf.tab_stops.add_tab_stop(Cm(21.0 - 3.0), WD_TAB_ALIGNMENT.RIGHT)
    r = p.add_run('Supplement to the presubmission enquiry\t22 November 2026')
    r.font.size = Pt(9)
    rule(doc)

    para(doc, 'Bridging protein and nucleic-acid pockets via shared physicochemical representations '
              'for 3D molecular generation with NA2smol \u2014 main figures and table', size=14, bold=True, space_before=4, space_after=2)
    para(doc, 'Letian Gao, Yi Li, Wenbiao Zhou, Jiahao Chen, Bing Su, Xing Li and Zhi John Lu', size=9, space_after=2)
    para(doc, 'Tsinghua University; Renmin University of China; Institute of Zoology, Chinese Academy of Sciences', size=9, italic=True, space_after=2)
    para(doc, 'Equal contribution: Letian Gao, Yi Li, Wenbiao Zhou and Jiahao Chen. Corresponding authors: Bing Su, Xing Li and Zhi John Lu.', size=8, space_after=8)
    para(doc, 'This supplement reproduces the five main figures and the main table with the captions as '
              'submitted with the manuscript.', size=9, italic=True, space_after=6)
    for item in ['Figure 1 \u2013 Workflow, training and candidate prioritisation',
                 'Figure 2 \u2013 Paired benchmark evaluation on the DUD-E protein benchmark and the RNA test set',
                 'Figure 3 \u2013 Comparison of the four NA2smol training configurations',
                 'Figure 4 \u2013 Structure-based design of fluorogenic ligands for the Clivia aptamer',
                 'Figure 5 \u2013 Experimental validation of a ligand candidate for the KRAS promoter G-quadruplex',
                 'Table 1 \u2013 Metric values for both benchmarks and the four training configurations']:
        para(doc, item, size=9, space_after=2)
    doc.add_page_break()

    # Render each figure from its authoritative PDF at build time. The PNG copies sitting beside
    # the PDFs are hand-made leftovers that go stale whenever a figure is regenerated; deriving
    # the image here means the Word file cannot drift away from the manuscript.
    png_dir = Path('/tmp/dsx_figs')
    png_dir.mkdir(parents=True, exist_ok=True)
    sources = [('Figure 1', ROOT / 'figures/figure1_nbt.pdf'),
               ('Figure 2', ROOT / 'figures/figure2_nbt.pdf'),
               ('Figure 3', ROOT / 'figures/figure3_nbt.pdf'),
               ('Figure 4', ROOT / 'figures/figure5_clivia.pdf'),
               ('Figure 5', ROOT / 'figures/figure5_kras.pdf')]
    figs = []
    for label, pdf in sources:
        out = png_dir / (pdf.stem + '.png')
        subprocess.run(['pdftoppm', '-r', '200', '-png', '-f', '1', '-l', '1',
                        '-singlefile', str(pdf), str(out.with_suffix(''))], check=True)
        figs.append((label, str(out), caps['fig' + label.split()[-1]]))
    for i, (label, img, cap) in enumerate(figs):
        if i:
            doc.add_page_break()
        heading(doc, label, before=0, after=4)
        pic = doc.add_picture(img, width=width)
        max_h = Cm(16.0)
        if pic.height > max_h:
            pic.height = max_h
            pic.width = int(pic.width * max_h / pic.height)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.space_after = Pt(6)
        para(doc, detex(cap), size=CAPTION_PT, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6)

    # tables, transcribed from the generated table sources
    doc.add_page_break()
    heading(doc, 'Table 1', before=0, after=4)
    para(doc, detex(caps['table1']), size=CAPTION_PT, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         space_after=6)
    add_radar_table(doc, width)

    out = ROOT / 'presubmission_supplement.docx'
    doc.save(out)
    return out


def add_radar_table(doc, width) -> None:
    """Table 1, from the generated LaTeX source so the numbers cannot drift."""
    tex = (ROOT / 'table_radar_values.tex').read_text()
    body = tex[tex.index(r'\midrule') + len(r'\midrule'):tex.index(r'\bottomrule')]
    rows = [r.strip() for r in body.split(r'\\') if r.strip()]
    parsed = []
    for row in rows:
        row = strip_rules(row)
        if row.startswith(r'\multicolumn'):
            parsed.append(('PANEL', multicolumn_text(row)))
            continue
        if not row:
            continue
        cells = [c.strip() for c in row.split('&')]
        parsed.append(('ROW', cells))

    header = ['Method', 'Dock.', 'RMSD', 'QED', 'SA', 'SEI', 'Strain', 'Rec.', 'Top-2']
    data_rows = sum(1 for k, _ in parsed if k == 'ROW')
    table = doc.add_table(rows=0, cols=len(header))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = table.add_row().cells
    for c, h in zip(hdr, header):
        c.text = ''
        run = c.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(7)

    for kind, payload in parsed:
        if kind == 'PANEL':
            row = table.add_row()
            merged = row.cells[0].merge(row.cells[-1])
            merged.text = ''
            run = merged.paragraphs[0].add_run(payload)
            run.italic = True
            run.font.size = Pt(7)
            continue
        row = table.add_row()
        for cell, raw in zip(row.cells, payload):
            bold = raw.startswith(r'\textbf') or raw.startswith(r'$\mathbf')
            under = raw.startswith(r'\underline')
            txt = detex(raw)
            cell.text = ''
            run = cell.paragraphs[0].add_run(txt)
            run.font.size = Pt(7)
            run.bold = bold
            run.underline = under
    # method column wide enough for "Pretrain+Constrain" on one line
    set_col_widths(table, [3.9] + [1.75] * 8)
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.space_before = Pt(0)


def add_overlap_table(doc, width) -> None:
    """Table 2: chemical overlap of the benchmark with the training data.

    Written out cell by cell rather than parsed from LaTeX, because the table stacks three
    panels with different column meanings and a parser would have to guess at the group rows.
    """
    rows = [
        ('head', ['Benchmark', 'n', 'Training source', 'Exact match, n (%)', 'ECFP4-TS \u22650.5, n (%)']),
        ('panel', 'A. Reference active ligands'),
        ('row', ['DUD-E', '35,237', 'Fine-tuning', '761 (2.2)', '12,818 (36.4)']),
        ('row', ['', '', 'Pretraining', '3,240 (9.2)', '31,722 (90.0)']),
        ('row', ['RNA test set', '50', 'Fine-tuning', '9 (18.0)', '29 (58.0)']),
        ('row', ['', '', 'Pretraining', '2 (4.0)', '36 (72.0)']),
        ('panel', 'B. NA2smol generated molecules'),
        ('row', ['DUD-E', '102,471', 'Fine-tuning', '80 (0.08)', '7,450 (7.3)']),
        ('row', ['', '', 'Pretraining', '365 (0.36)', '26.5 (25.2-27.9)\u2020']),
        ('row', ['RNA test set', '34,834', 'Fine-tuning', '57 (0.16)', '2,087 (6.0)']),
        ('row', ['', '', 'Pretraining', '153 (0.44)', '16.8 (15.8-17.9)\u2020']),
        ('panel', 'C. NA2smol performance after excluding potentially leaky targets'),
        ('row', ['DUD-E', '84', 'All eligible targets', '\u2014', '0.692']),
        ('row', ['', '84', 'Excluding FT-similar actives', '\u2014', '0.578']),
        ('row', ['', '81', 'Excluding PT-similar actives', '\u2014', '0.238']),
        ('row', ['RNA test set', '27', 'All eligible targets', '\u2014', '0.218']),
        ('row', ['', '18', 'Excluding FT-similar actives', '\u2014', '0.249']),
        ('row', ['', '14', 'Excluding PT-similar actives', '\u2014', '0.052']),
    ]
    table = doc.add_table(rows=0, cols=5)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for kind, payload in rows:
        if kind == 'panel':
            row = table.add_row()
            merged = row.cells[0].merge(row.cells[-1])
            merged.text = ''
            run = merged.paragraphs[0].add_run(payload)
            run.italic = True
            run.font.size = Pt(8)
            continue
        row = table.add_row()
        for cell, txt in zip(row.cells, payload):
            cell.text = ''
            run = cell.paragraphs[0].add_run(txt)
            run.font.size = Pt(8)
            run.bold = kind == 'head'
    set_col_widths(table, [3.4, 1.9, 4.4, 3.4, 4.2])
    for row in table.rows:
        for cell in row.cells:
            for para_ in cell.paragraphs:
                para_.paragraph_format.space_after = Pt(0)
                para_.paragraph_format.space_before = Pt(0)
    note = doc.add_paragraph()
    r = note.add_run('\u2020Estimated from a stratified sample of the pretraining library '
                     '(n=4,217 DUD-E, n=4,729 RNA test set). Panel C recovery is the mean '
                     'target-recovery proportion over ECFP4-Tanimoto thresholds 0.25-0.50, '
                     'recomputed on the surviving targets.')
    r.font.size = Pt(7)
    note.paragraph_format.space_before = Pt(3)


if __name__ == '__main__':
    # The enquiry is the author-maintained Word letter; only the supplement is regenerated here.
    print('wrote', build_supplement())
