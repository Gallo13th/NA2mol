"""Build journal-specific Word versions of the presubmission enquiry letter.

The letter itself is author-polished and lives in `presubmission_enquiry.docx`; this script does
not regenerate it, it clones it and swaps only the journal-facing sentences, so the typography,
margins and paragraph formatting stay exactly as approved.

Two variants are produced:
  presubmission_enquiry_nature_methods.docx        (Nature Methods)
  presubmission_enquiry_nature_biotechnology.docx  (Nature Biotechnology)

Difference between the two, deliberately kept small and auditable:
  * the journal named in the closing paragraph;
  * for Nature Biotechnology only, the closing sentence of the scope paragraph, which
    foregrounds the validated end-to-end discovery pipeline (the Nature Methods version keeps
    the approved wording of the current letter verbatim).

Usage: python scripts/build_enquiry_variants.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'presubmission_enquiry.docx'

# The scope paragraph is paragraph index 12 and the closing request index 13 in the source
# letter. Both are asserted below so a re-flow of the source letter fails loudly instead of
# silently editing the wrong paragraph.
SCOPE_MARKER = 'We believe these advances will be of broad interest'
CLOSING_MARKER = 'We would appreciate your assessment'

VARIANTS = {
    'nature_methods': {
        'journal': 'Nature Methods',
        'file': 'presubmission_enquiry_nature_methods.docx',
        # unchanged from the approved letter: the source already targets Nature Methods
        'scope': None,
    },
    'nature_biotechnology': {
        'journal': 'Nature Biotechnology',
        'file': 'presubmission_enquiry_nature_biotechnology.docx',
        'scope': ('By enabling cross-domain knowledge transfer and complex-independent density '
                  'pretraining, NA2smol extends pocket-conditioned molecular generation beyond '
                  'protein targets, offering a new route to overcoming the structural data '
                  'bottleneck in nucleic-acid-targeted drug design. Together with systematic '
                  'benchmarking and experimental validation, this work establishes a working '
                  'design pipeline with its benchmark and experimental endpoint, rather than '
                  'only a computational model. We believe these advances will be of interest to '
                  'readers working on nucleic-acid therapeutics, computational drug discovery '
                  'and generative molecular design, for whom a validated end-to-end pipeline '
                  '\u2014 from pocket-conditioned generation through computational prioritisation '
                  'to an experimentally confirmed small molecule \u2014 is directly enabling.'),
    },
}


def find_paragraph(doc, marker: str):
    for idx, para in enumerate(doc.paragraphs):
        if marker in para.text:
            return idx, para
    raise SystemExit(f'marker not found in the source letter: {marker!r}')


def set_paragraph_text(para, text: str) -> None:
    """Replace a single-run paragraph without disturbing its formatting."""
    if len(para.runs) != 1:
        raise SystemExit(f'expected a single run, found {len(para.runs)}: {para.text[:60]!r}')
    para.runs[0].text = text


def build(name: str, spec: dict) -> Path:
    doc = Document(SOURCE)
    _, scope = find_paragraph(doc, SCOPE_MARKER)
    _, closing = find_paragraph(doc, CLOSING_MARKER)

    if spec['scope'] is not None:
        set_paragraph_text(scope, spec['scope'])
    closing_text = closing.text
    if 'Nature Methods' not in closing_text:
        raise SystemExit(f'closing paragraph has no journal name to replace: {closing_text!r}')
    set_paragraph_text(closing, closing_text.replace('Nature Methods', spec['journal']))

    out = ROOT / spec['file']
    doc.save(out)
    return out


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit(f'source letter missing: {SOURCE}')
    for name, spec in VARIANTS.items():
        out = build(name, spec)
        print(f'wrote {out.relative_to(ROOT)}  ({spec["journal"]})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
