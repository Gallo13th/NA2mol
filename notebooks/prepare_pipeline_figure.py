#!/usr/bin/env python
"""Prepare the cleaned workflow figure used as manuscript Fig. 1."""

from __future__ import annotations

from pathlib import Path
import re

import cairosvg
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reproduction" / "figures"
SRC = FIG_DIR / "new_figure1_pipeline.svg"
OUT = FIG_DIR / "new_figure1_pipeline_clean"


def scale_text(match: re.Match[str]) -> str:
    size = float(match.group(1))
    if size >= 100:
        size *= 0.82
    return f'font-size="{size:.1f}"'


def clean_svg(svg: str) -> str:
    svg = re.sub(r'font-size="([0-9.]+)"', scale_text, svg)
    svg = re.sub(r">RNA(?=<tspan)", ">NA", svg)
    replacements = {
        "PRIORITIASTION": "PRIORITISATION",
        'x="504.167" y="0">PREPARATION': 'x="565" y="0">PREPARATION',
        'x="641.667" y="0">TRAINING': 'x="700" y="0">TRAINING',
        'x="1145.83" y="0">&amp;': 'x="1220" y="0">&amp;',
        'x="1310.83" y="0">PRIORITISATION': 'x="1415" y="0">PRIORITISATION',
        'x="-854.491" y="-990">Dataset': 'x="-770" y="-990">Dataset',
        'x="-745.002" y="412">Dataset': 'x="-655" y="412">Dataset',
        'x="-710.189" y="1836">Dataset': 'x="-615" y="1836">Dataset',
        'x="-149.722" y="-1246">Screen': 'x="-95" y="-1246">Screen',
        'x="379.653" y="-1246">Tunnel': 'x="490" y="-1246">Tunnel',
        'x="379.271" y="0">in': 'x="420" y="0">in',
        'x="309.076" y="132">Library': 'x="365" y="132">Library',
        'x="-599.708" y="668">Candidate': 'x="-535" y="668">Candidate',
    }
    for old, new in replacements.items():
        svg = svg.replace(old, new)
    return svg


def main() -> None:
    svg = clean_svg(SRC.read_text(errors="ignore"))
    svg_path = OUT.with_suffix(".svg")
    svg_path.write_text(svg)

    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(OUT.with_suffix(".pdf")))
    cairosvg.svg2ps(bytestring=svg.encode(), write_to=str(OUT.with_suffix(".eps")))
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT.with_suffix(".png")), output_width=4200)

    with Image.open(OUT.with_suffix(".png")) as img:
        img.save(OUT.with_suffix(".tiff"), dpi=(600, 600))

    print(f"Wrote {OUT}.svg/.pdf/.eps/.png/.tiff")


if __name__ == "__main__":
    main()
