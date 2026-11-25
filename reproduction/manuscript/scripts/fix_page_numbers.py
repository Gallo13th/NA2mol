#!/usr/bin/env python3
"""Repair page numbers that sn-jnl prints inside the last line of a page.

When a page ends inside a float caption the class appends its footer number to that horizontal
line instead of placing it in the footer box, which leaves e.g. a "10" printed on top of a caption
word. This script finds those runs (the page number drawn with an upward offset) and re-emits the
number at the footer position; pages whose number is already correct are left untouched.
"""
import re
import sys

import pikepdf

RUN = re.compile(
    rb"(?P<lead>0 g 0 G\s*0 g 0 G\s*)/F63 (?P<size>[\d.]+) Tf (?P<dx>-?[\d.]+) (?P<dy>-?[\d.]+) Td "
    rb"\[\((?P<num>\d+)\)\]TJ\s*0 g 0 G"
)


def footer(num: bytes, width: float, height: float) -> bytes:
    x = width / 2.0 - 6.0
    y = 25.0
    return (b"BT /F63 9.9626 Tf 1 0 0 1 %0.2f %0.2f Tm [(%s)]TJ ET" % (x, y, num))


def main(path: str) -> int:
    pdf = pikepdf.open(path, allow_overwriting_input=True)
    fixed = 0
    for page in pdf.pages:
        box = page.MediaBox
        width = float(box[2]) - float(box[0])
        height = float(box[3]) - float(box[1])
        data = page.Contents.read_bytes()
        m = None
        for m in RUN.finditer(data):
            pass
        if m is None:
            continue
        dy = float(m.group("dy"))
        if dy <= 0:
            continue  # number already lands in the footer
        new = data[: m.start()] + data[m.end():]
        new += b"\n" + footer(m.group("num"), width, height) + b"\n"
        page.Contents.write(new)
        fixed += 1
    pdf.save(path)
    print(f"{path}: repaired {fixed} misplaced page number(s)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: fix_page_numbers.py <pdf>")
    raise SystemExit(main(sys.argv[1]))
