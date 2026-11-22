#!/usr/bin/env python3
"""Extract text from a Chrome-printed PDF (OpenReview page export).

Parses indirect objects, per-page font resources, ToUnicode CMaps, and
reconstructs lines from glyph positioning.
"""
import re, sys, zlib

SRC, OUT = sys.argv[1], sys.argv[2]
data = open(SRC, "rb").read()

objs = {}
for m in re.finditer(rb"(\d+)\s+0\s+obj(.*?)endobj", data, re.S):
    objs[int(m.group(1))] = m.group(2)

def get_obj(n):
    return objs.get(int(n), b"")

def get_stream(n):
    body = get_obj(n)
    m = re.search(rb"stream\r?\n(.*?)\r?\nendstream", body, re.S)
    if not m:
        return None
    raw = m.group(1)
    if b"/FlateDecode" in body:
        try:
            return zlib.decompress(raw)
        except Exception:
            return None
    return raw

# ---------- page tree ----------
def kids(n):
    b = get_obj(n)
    m = re.search(rb"/Kids\s*\[(.*?)\]", b, re.S)
    return [int(x) for x in re.findall(rb"(\d+)\s+0\s+R", m.group(1))] if m else []

root = None
for i, b in objs.items():
    if re.search(rb"/Type\s*/Pages", b) and not re.search(rb"/Parent", b):
        root = i
        break
page_ids = []
def walk(n):
    b = get_obj(n)
    if re.search(rb"/Type\s*/Page\b", b) and not re.search(rb"/Type\s*/Pages", b):
        page_ids.append(n)
        return
    for k in kids(n):
        walk(k)
if root is not None:
    walk(root)
if not page_ids:
    page_ids = [i for i, b in objs.items() if re.search(rb"/Type\s*/Page\b", b)]

# ---------- CMap ----------
def parse_cmap(n):
    s = get_stream(n)
    if not s:
        return {}
    mp = {}
    for blk in re.findall(rb"beginbfchar(.*?)endbfchar", s, re.S):
        for src, dst in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
            try:
                mp[int(src, 16)] = bytes.fromhex(dst.decode()).decode("utf-16-be", "replace")
            except Exception:
                pass
    for blk in re.findall(rb"beginbfrange(.*?)endbfrange", s, re.S):
        for m in re.finditer(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\[(.*?)\]", blk, re.S):
            lo, hi = int(m.group(1), 16), int(m.group(2), 16)
            dsts = re.findall(rb"<([0-9A-Fa-f]+)>", m.group(3))
            for i, d in enumerate(dsts):
                if lo + i <= hi:
                    try:
                        mp[lo + i] = bytes.fromhex(d.decode()).decode("utf-16-be", "replace")
                    except Exception:
                        pass
        for m in re.finditer(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(?=\n|$|endbfrange)", blk):
            lo, hi = int(m.group(1), 16), int(m.group(2), 16)
            try:
                base = int(m.group(3), 16)
                ln = max(2, len(m.group(3)) // 4)
                for c in range(lo, hi + 1):
                    mp[c] = (base + c - lo).to_bytes(ln, "big").decode("utf-16-be", "replace")
            except Exception:
                pass
    return mp

def page_fonts(page_body):
    res = page_body
    rm = re.search(rb"/Resources\s+(\d+)\s+0\s+R", page_body)
    if rm:
        res = get_obj(int(rm.group(1)))
    fm = re.search(rb"/Font\s*<<(.*?)>>", res, re.S)
    if not fm:
        return {}
    out = {}
    for name, oid in re.findall(rb"/([A-Za-z0-9]+)\s+(\d+)\s+0\s+R", fm.group(1)):
        fo = get_obj(int(oid))
        tu = re.search(rb"/ToUnicode\s+(\d+)\s+0\s+R", fo)
        two = bool(re.search(rb"/Subtype\s*/Type0", fo))
        out[name.decode()] = (parse_cmap(int(tu.group(1))) if tu else {}, two)
    return out

# ---------- content ----------
TOKEN = re.compile(
    rb"\s*(?:(?P<num>[-+]?(?:\d+\.\d*|\.\d+|\d+))"
    rb"|(?P<name>/[^\s/\[\]()<>{}%]*)"
    rb"|(?P<str>\((?:[^()\\]|\\.)*\))"
    rb"|(?P<hex><[0-9A-Fa-f\s]*>)"
    rb"|(?P<arr>\[)|(?P<arrend>\])"
    rb"|(?P<op>[A-Za-z'\"*]+))", re.S)

ESC = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b",
       b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\"}

def unescape(s):
    out = bytearray(); i = 0
    while i < len(s):
        c = s[i:i+1]
        if c == b"\\" and i + 1 < len(s):
            nx = s[i+1:i+2]
            if nx in ESC:
                out += ESC[nx]; i += 2; continue
            if nx in b"01234567":
                j = i + 1; o = b""
                while j < len(s) and len(o) < 3 and s[j:j+1] in b"01234567":
                    o += s[j:j+1]; j += 1
                out.append(int(o, 8) & 0xFF); i = j; continue
            if nx in (b"\r", b"\n"):
                i += 2; continue
            out += nx; i += 2; continue
        out += c; i += 1
    return bytes(out)

def decode_str(raw, font):
    if not font:
        return ""
    cmap, two = font
    out = []
    if two:
        for i in range(0, len(raw) - 1, 2):
            out.append(cmap.get((raw[i] << 8) | raw[i+1], ""))
    else:
        for b in raw:
            out.append(cmap.get(b, ""))
    return "".join(out)

def process(content, fonts, runs):
    x = y = 0.0
    a, d = 1.0, 1.0
    leading = 0.0
    cur = None
    size = 12.0
    stack = []
    i = 0
    while True:
        m = TOKEN.search(content, i)
        if not m:
            break
        i = m.end()
        k = m.lastgroup
        if k == "num":
            stack.append(float(m.group("num"))); continue
        if k == "name":
            stack.append(("name", m.group("name"))); continue
        if k == "str":
            stack.append(("str", unescape(m.group("str")[1:-1]))); continue
        if k == "hex":
            h = re.sub(rb"\s", b"", m.group("hex")[1:-1])
            if len(h) % 2: h += b"0"
            try:
                stack.append(("str", bytes.fromhex(h.decode())))
            except Exception:
                pass
            continue
        if k == "arr":
            stack.append(("arrstart", None)); continue
        if k == "arrend":
            items = []
            while stack and not (isinstance(stack[-1], tuple) and stack[-1][0] == "arrstart"):
                items.append(stack.pop())
            if stack: stack.pop()
            stack.append(("arr", list(reversed(items))))
            continue
        op = m.group("op").decode("latin-1")
        try:
            if op == "Tf":
                for it in stack:
                    if isinstance(it, tuple) and it[0] == "name":
                        cur = fonts.get(it[1].decode("latin-1"))
                for it in reversed(stack):
                    if isinstance(it, float):
                        size = it; break
            elif op == "Tm":
                a, d = stack[-6], stack[-3]
                x, y = stack[-2], stack[-1]
            elif op in ("Td", "TD"):
                tx, ty = stack[-2], stack[-1]
                if op == "TD": leading = -ty
                x += tx * a; y += ty * d
            elif op == "T*":
                y -= leading
            elif op == "TL":
                leading = stack[-1]
            elif op in ("Tj", "'", '"'):
                for it in reversed(stack):
                    if isinstance(it, tuple) and it[0] == "str":
                        t = decode_str(it[1], cur)
                        if t.strip():
                            runs.append((x, y, t, size))
                        break
            elif op == "TJ":
                if stack and isinstance(stack[-1], tuple) and stack[-1][0] == "arr":
                    parts = []
                    for it in stack[-1][1]:
                        if isinstance(it, tuple) and it[0] == "str":
                            parts.append(decode_str(it[1], cur))
                        elif isinstance(it, float) and it <= -100:
                            parts.append(" ")
                    t = "".join(parts)
                    if t.strip():
                        runs.append((x, y, t, size))
        except (IndexError, TypeError):
            pass
        stack.clear()

all_lines = []
for pi, pid in enumerate(page_ids):
    body = get_obj(pid)
    fonts = page_fonts(body)
    content = b""
    ca = re.search(rb"/Contents\s*\[(.*?)\]", body, re.S)
    cm = re.search(rb"/Contents\s+(\d+)\s+0\s+R", body)
    if ca:
        for ref in re.findall(rb"(\d+)\s+0\s+R", ca.group(1)):
            s = get_stream(int(ref))
            if s: content += s + b"\n"
    elif cm:
        s = get_stream(int(cm.group(1)))
        if s: content = s
    runs = []
    try:
        process(content, fonts, runs)
    except Exception:
        pass
    runs.sort(key=lambda r: (-round(r[1]), r[0]))
    lines, cur_y, cur, last_x, cur_sz = [], None, "", None, 12.0
    for (rx, ry, t, sz) in runs:
        if cur_y is None or abs(ry - cur_y) > 3.0:
            if cur: lines.append(cur)
            cur, cur_y, last_x, cur_sz = t, ry, rx, sz
        else:
            if last_x is not None and rx - last_x > max(2.0, 1.2 * max(cur_sz, sz)):
                cur += " "
            cur += t
            last_x, cur_sz = rx, max(cur_sz, sz)
    if cur: lines.append(cur)
    all_lines.append("===== PAGE %d =====" % (pi + 1))
    all_lines.extend(lines)

open(OUT, "w", encoding="utf-8").write("\n".join(all_lines))
print("pages:", len(page_ids), "lines:", len(all_lines), "->", OUT)
