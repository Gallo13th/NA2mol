"""Word versions of the two cover letters, matching the PDF layout."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent
COMMIT = 'bac48aa'

P1 = ('Structure-based molecular generation has become routine for protein pockets, but nucleic '
      'acids have been left behind, for two reasons. Part of the gap is data: curated '
      'nucleic-acid\u2013small-molecule resources hold hundreds of non-redundant entries against the '
      'millions of protein complexes available for protein training. Part of it is representation. '
      'Most pocket-conditioned generators encode a pocket through amino-acid-specific features for '
      'which there is no nucleotide analogue, so a model trained on proteins cannot simply be '
      'pointed at an RNA site.')

P2 = ('We report NA2smol, a Bayesian flow network generator that addresses both constraints '
      'together. The pocket representation is restricted to descriptors that proteins and nucleic '
      'acids share: pharmacophore type, hybridisation state, van der Waals radius and partial '
      'charge. Structural supervision comes from ligand-only electron-density pretraining, and a '
      'teacher\u2013student alignment loss, with a cross-view contrastive term completing the '
      'objective. One model generates three-dimensional ligands for either class of binding '
      'site.')

P3 = ('We benchmark against six structure-based generators spanning the main generation '
      'paradigms: MolCRAFT, Pocket2Mol and ED2Mol, together with DiffSBDD, GraphBP and FLAG. '
      'On the RNA test set, NA2smol retained 239 molecules across 17 of 27 evaluable '
      'targets under the most stringent tier of a four-stage screening cascade, 0.68% of 35,358 '
      'valid generations, and the protein-trained baselines retained far fewer. On retrospective '
      'ECFP4 recovery, NA2smol leads both the RNA and protein benchmarks among these generators. '
      'The same output, carried through post-processing '
      'of the generated library, produced a bis-benzothiazole that reduced normalised '
      'thioflavin T fluorescence in a concentration-dependent manner, with an apparent '
      'half-maximal concentration of 22.4 \u00b5M in a KRAS promoter G-quadruplex assay.')

P4 = ('The paper reports where NA2smol leads and where it does not. On conformational strain it is '
      'mid-pack rather than leading, and the strain table reports success separately from the pass '
      'rate so that geometries which could not be evaluated are visible as such. The experimental '
      'endpoint rests on one compound selected through an integrated screen, which is what a first '
      'prospective test of a generative pipeline can offer.')

FIT = {
    'natmethods': (
        'We believe the work suits Nature Methods because the enabling step it describes is a '
        'method one, and the standard it holds itself to is use rather than a single benchmark '
        'result: a representation that transfers between two chemically different target classes, '
        'a training strategy that is specified in full, and a benchmark run against six external '
        'generators through one downstream pipeline. The complete training code, '
        'generated-molecule tables, screening summaries and figure source data are available at '
        'https://github.com/Gallo13th/NA2mol (commit ' + COMMIT + '), and the manuscript states '
        'every target universe and yield denominator so the counts can be recomputed.'
    ),
    'ncs': (
        'We believe the work suits Nature Computational Science because its contribution is a '
        'computational design question rather than a single application: how to build a '
        'representation and a training strategy that transfer across two chemically different '
        'target classes when labelled structural data exist for only one. Nucleic-acid pockets '
        'are a demanding test of that question, and the paper reports the pipeline end to end. '
        'The complete training code, generated-molecule tables, screening summaries and figure '
        'source data are available at https://github.com/Gallo13th/NA2mol (commit ' + COMMIT + '), '
        'and the manuscript states the target universes and yield denominators explicitly so '
        'that the reported counts can be recomputed.'
    ),
}

CLOSE = ('The manuscript is original, is not under consideration elsewhere, and all authors have '
         'approved this submission. We declare no competing interests. We would be glad to '
         'suggest referees in generative molecular design, nucleic-acid recognition and '
         'structure-based screening.')

JOURNALS = {'natmethods': 'Nature Methods', 'ncs': 'Nature Computational Science'}


def para(doc, text='', size=10, bold=False, italic=False, before=0, after=5, align=None):
    p = doc.add_paragraph()
    if text:
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.bold = bold
        r.italic = italic
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if align is not None:
        p.alignment = align
    return p


def build(key):
    doc = Document()
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.top_margin, s.bottom_margin = Cm(1.9), Cm(1.9)
    s.left_margin, s.right_margin = Cm(2.3), Cm(2.3)
    st = doc.styles['Normal']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(10)
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts')
        rpr.append(rf)
    rf.set(qn('w:eastAsia'), 'Times New Roman')

    para(doc, '10 November 2026', after=10, align=None)
    para(doc, 'The Editors', after=0, align=None)
    para(doc, JOURNALS[key], italic=True, after=12, align=None)
    para(doc, 'Re: submission of \u201cBridging protein and nucleic-acid pockets via shared '
              'physicochemical representations for 3D molecular generation with NA2smol\u201d', bold=True, after=8, align=None)
    para(doc, 'Dear Editors,', after=6, align=None)

    for block in (P1, P2, P3, P4, FIT[key], CLOSE):
        para(doc, block, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    para(doc, 'Thank you for considering our work.', after=12, align=None)
    para(doc, 'Yours sincerely,', after=16, align=None)
    para(doc, 'Zhi John Lu', after=0, align=None)
    para(doc, 'Associate Professor, School of Life Sciences, Tsinghua University', after=0, align=None)
    para(doc, 'on behalf of all authors', after=0, align=None)
    para(doc, 'zhilu@tsinghua.edu.cn', after=0, align=None)

    out = ROOT / f'cover_letter_{key}.docx'
    doc.save(out)
    return out


if __name__ == '__main__':
    for k in JOURNALS:
        print('wrote', build(k))
