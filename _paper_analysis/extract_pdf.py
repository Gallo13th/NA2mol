#!/usr/bin/env python3
"""Minimal pdfTeX-oriented text extractor: walks content streams, tracks text
matrix, rebuilds lines by (y, x) ordering, maps T1 ligature codes."""
import re, sys, zlib, os

SRC = sys.argv[1]
OUT = sys.argv[2]

data = open(SRC, "rb").read()

# ---------- content stream extraction (handle objects, skip compressed oddities)
streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S)
pages = []
for s in streams:
    try:
        d = zlib.decompress(s)
    except Exception:
        d = s  # maybe uncompressed
    if b"BT" in d:
        pages.append(d)

# ---------- tokenizer
TOKEN = re.compile(
    rb"\s*(?:"
    rb"(?P<num>[-+]?(?:\d+\.\d*|\.\d+|\d+))"
    rb"|(?P<name>/[^\s/\[\]()<>{}%]*)"
    rb"|(?P<str>\((?:[^()\\]|\\.)*\))"
    rb"|(?P<hex><[0-9A-Fa-f\s]*>)"
    rb"|(?P<arr>\[)|(?P<arrend>\])"
    rb"|(?P<dict><<)|(?P<dictend>>>)"
    rb"|(?P<op>[A-Za-z'\"*]+)"
    rb"|(?P<punct>[{}])"
    rb")", re.S)

ESC = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b",
       b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\"}

def unescape(s):
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i:i+1]
        if c == b"\\" and i + 1 < len(s):
            nxt = s[i+1:i+2]
            if nxt in ESC:
                out += ESC[nxt]; i += 2; continue
            if nxt.isdigit() and nxt <= b"7":
                j = i + 1
                oct_ = b""
                while j < len(s) and len(oct_) < 3 and s[j:j+1] in b"01234567":
                    oct_ += s[j:j+1]; j += 1
                out.append(int(oct_, 8) & 0xFF); i = j; continue
            if nxt == b"\r":
                i += 2
                if s[i:i+1] == b"\n": i += 1
                continue
            if nxt == b"\n":
                i += 2; continue
            out += nxt; i += 2; continue
        out += c; i += 1
    return bytes(out)

LIG = {2: "fi", 3: "fl", 4: "ff", 5: "ffi", 6: "ffl"}
SPECIAL = {0x16: "\u2013", 0x97: "\u2014", 0x91: "\u2018", 0x92: "\u2019",
           0x93: "\u201c", 0x94: "\u201d", 0x85: "\u2026", 0xA0: " ",
           0x13: "\u2013", 0x14: "\u2014"}

def decode_text(b):
    out = []
    for ch in b:
        if ch in LIG:
            out.append(LIG[ch])
        elif ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch < 32:
            out.append("")
        elif ch < 127:
            out.append(chr(ch))
        else:
            out.append(chr(ch))
    return "".join(out)

# ---------- interpreter
runs = []  # (x, y, text) per page

def process(page_bytes, page_runs):
    x = y = 0.0
    leading = 0.0
    m_a, m_b, m_c, m_d = 1.0, 0.0, 0.0, 1.0
    stack = []
    i = 0
    s = page_bytes
    while True:
        m = TOKEN.search(s, i)
        if not m:
            break
        i = m.end()
        kind = m.lastgroup
        if kind == "num":
            stack.append(float(m.group("num")))
            continue
        if kind == "name":
            stack.append(("name", m.group("name")))
            continue
        if kind == "str":
            stack.append(("str", unescape(m.group("str")[1:-1])))
            continue
        if kind == "hex":
            h = re.sub(rb"\s", b"", m.group("hex")[1:-1])
            if len(h) % 2: h += b"0"
            stack.append(("str", bytes.fromhex(h.decode())))
            continue
        if kind == "arr":
            stack.append(("arrstart", None)); continue
        if kind == "arrend":
            items = []
            while stack and not (isinstance(stack[-1], tuple) and stack[-1][0] == "arrstart"):
                items.append(stack.pop())
            if stack: stack.pop()
            stack.append(("arr", list(reversed(items))))
            continue
        if kind in ("dict", "dictend"):
            continue
        if kind == "punct":
            continue
        # operator
        op = m.group("op").decode("latin-1")
        try:
            if op == "BT":
                x = y = 0.0
            elif op == "Tm":
                m_a, m_b, m_c, m_d = stack[-6], stack[-5], stack[-4], stack[-3]
                x, y = stack[-2], stack[-1]
            elif op in ("Td", "TD"):
                tx, ty = stack[-2], stack[-1]
                if op == "TD": leading = -ty
                x += tx * m_a
                y += ty * m_d
            elif op == "T*":
                y -= leading
            elif op == "TL":
                leading = stack[-1]
            elif op == "Tj":
                if stack and stack[-1][0] == "str":
                    txt = decode_text(stack[-1][1])
                    if txt.strip(): page_runs.append((x, y, txt))
            elif op == "TJ":
                if stack and stack[-1][0] == "arr":
                    parts = []
                    for it in stack[-1][1]:
                        if isinstance(it, tuple) and it[0] == "str":
                            parts.append(decode_text(it[1]))
                        elif isinstance(it, float):
                            if it <= -100:
                                parts.append(" ")
                            elif it <= -30:
                                parts.append("")
                    txt = "".join(parts)
                    if txt.strip(): page_runs.append((x, y, txt))
        except IndexError:
            pass
        stack.clear()

all_lines = []
for pi, page in enumerate(pages):
    page_runs = []
    process(page, page_runs)
    # group by y
    page_runs.sort(key=lambda r: (-round(r[1]), r[0]))
    lines = []
    cur_y = None
    cur = []
    for (rx, ry, txt) in page_runs:
        if cur_y is None or abs(ry - cur_y) > 2.5:
            if cur: lines.append(" ".join(cur))
            cur = [txt]; cur_y = ry
        else:
            cur.append(txt)
    if cur: lines.append(" ".join(cur))
    all_lines.append("===== PAGE %d =====" % (pi + 1))
    all_lines.extend(lines)

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(all_lines))
print("pages:", len(pages), "lines:", len(all_lines), "->", OUT)
