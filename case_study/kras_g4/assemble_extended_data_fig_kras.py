"""Compose the Extended Data figure holding the relaxed-product and
ensemble-triage scatters moved out of Figure 4.

Output: reproduction/figures/ed_figure4.{pdf,png}
"""
from pathlib import Path

import fitz
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
PANELS = BASE / "fig5_panels"
OUT = Path(__file__).resolve().parents[2] / "reproduction" / "figures"

PAGE_W_IN, PAGE_H_IN = 7.0, 3.6
W, H = PAGE_W_IN * 72, PAGE_H_IN * 72
LAYOUT = {
    "c_relaxed": (0.015, 0.03, 0.495, 0.97),
    "d_triage":  (0.505, 0.03, 0.985, 0.97),
}


def place(page, pdf, slot):
    x0, y0, x1, y1 = slot
    rect = fitz.Rect(x0 * W, (1 - y1) * H, x1 * W, (1 - y0) * H)
    src = fitz.open(str(pdf)); s = src[0].rect
    scale = min(rect.width / s.width, rect.height / s.height)
    dw, dh = s.width * scale, s.height * scale
    cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
    page.show_pdf_page(fitz.Rect(cx - dw / 2, cy - dh / 2, cx + dw / 2, cy + dh / 2), src, 0)


def main():
    out = fitz.open()
    page = out.new_page(width=W, height=H)
    for name, slot in LAYOUT.items():
        place(page, PANELS / f"{name}.pdf", slot)
    OUT.mkdir(parents=True, exist_ok=True)
    page.parent.save(str(OUT / "ed_figure4.pdf"))
    page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(str(OUT / "ed_figure4.png"))
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    svg = ET.Element(f"{{{namespace}}}svg", {"width": f"{W}pt", "height": f"{H}pt",
                                            "viewBox": f"0 0 {W} {H}"})
    for name, slot in LAYOUT.items():
        source = ET.parse(PANELS / f"{name}.svg").getroot()
        source_width, source_height = [float(source.get(key).removesuffix("pt")) for key in ("width", "height")]
        left, bottom, right, top = slot
        scale = min((right - left) * W / source_width, (top - bottom) * H / source_height)
        source.set("x", str((left + right) * W / 2 - source_width * scale / 2))
        source.set("y", str((2 - top - bottom) * H / 2 - source_height * scale / 2))
        source.set("width", str(source_width * scale))
        source.set("height", str(source_height * scale))
        for element in source.iter():
            if "id" in element.attrib:
                element.set("id", name + "_" + element.get("id"))
            for key, value in list(element.attrib.items()):
                value = value.replace("url(#", f"url(#{name}_")
                if value.startswith("#") and key.endswith("href"):
                    value = "#" + name + "_" + value[1:]
                element.set(key, value)
        svg.append(source)
    ET.ElementTree(svg).write(OUT / "ed_figure4.svg", encoding="utf-8", xml_declaration=True)
    print("wrote", OUT / "ed_figure4.pdf")


if __name__ == "__main__":
    main()
