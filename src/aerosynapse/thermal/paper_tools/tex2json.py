"""Convert a paper's LaTeX sources (the subset used in paper/paper*) into a JSON document model for the Word builder.

    python -m aerosynapse.thermal.paper_tools.tex2json paper/paper2        -> paper/paper2/build/doc.json

Handles: frontmatter (title, author placeholders, highlights, abstract, keywords, running title from
\\runningtitle), nomenclature and glossary lists (environments nomenclature / keyterms: \\item[term] definition,
rendered as two-column tables; an item without a definition is a group heading), sections / subsections /
paragraphs (numbered; appendix lettered), equations (numbered, converted to Word linear math 'UnicodeMath'), inline
math, figures (PNG previews of the generated PDFs), tables (\\input rows, booktabs rules), itemize / enumerate,
\\cite (numbered in order of first citation, as elsarticle-num), \\ref / \\eqref, accents and typographic ligatures,
and the generated number macros of numbers.tex (expanded). Anything not recognised is reported, not silently dropped.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
WARN: list[str] = []

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε", "theta": "θ",
    "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "phi": "ϕ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
MATHSYM = {
    "partial": "∂", "nabla": "∇", "cdot": "⋅", "cdots": "⋯", "ldots": "…", "dots": "…", "times": "×", "pm": "±",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "approx": "≈", "infty": "∞", "to": "→", "rightarrow": "→",
    "top": "⊤", "prime": "′", "circ": "∘", "sim": "∼", "propto": "∝", "in": "∈", "ne": "≠", "neq": "≠",
    "qquad": "\u2003\u2003", "quad": "\u2003", ",": "\u2009", ";": "\u2005", ":": "\u2005", "!": "", " ": " ",
    "%": "%", "ln": "ln ", "exp": "exp ", "log": "log ", "max": "max ", "min": "min ", "lvert": "|", "rvert": "|",
}
ACCENT = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303", "c": "\u0327"}
TEXTSYM = {"%": "%", "&": "&", "_": "_", "#": "#", "$": "$", "{": "{", "}": "}", ",": "\u2009", ";": "\u2005",
           " ": " ", "O": "Ø", "o": "ø", "dots": "…", "ldots": "…", "S": "§", "textendash": "–", "textemdash": "—"}


# ----------------------------------------------------------------------------------------------- source handling
def strip_comments(s: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in s.splitlines())


def read(path: Path) -> str:
    return strip_comments(path.read_text(encoding="utf-8"))


def group(s: str, i: int) -> tuple[str, int]:
    """Content of the brace group starting at s[i] == '{' and the index after it."""
    assert s[i] == "{", s[i : i + 40]
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
        j += 1
    raise ValueError("unbalanced braces near: " + s[i : i + 60])


def macros(paper: Path) -> dict[str, str]:
    out = {}
    for f in (paper / "numbers.tex", paper / "main.tex"):
        s = read(f)
        for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}", s):
            body, _ = group(s, m.end())
            out[m.group(1)] = body
    return out


def expand(s: str, mac: dict[str, str]) -> str:
    pat = re.compile(r"\\(" + "|".join(sorted(mac, key=len, reverse=True)) + r")(?![A-Za-z])(\{\})?")
    for _ in range(5):
        t = pat.sub(lambda m: mac[m.group(1)].replace("\\", "\\\\") if False else mac[m.group(1)], s)
        if t == s:
            break
        s = t
    return s


def inline_inputs(s: str, base: Path) -> str:
    def rep(m):
        f = base / (m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))
        return inline_inputs(read(f), base) if f.exists() else ""

    return re.sub(r"\\input\{([^}]+)\}", rep, s)


# ----------------------------------------------------------------------------------------------- math
def math_plain(s: str) -> str:
    """Math as plain Unicode text (section headings, plain-text fields)."""
    s = re.sub(r"\\(mathrm|text|boldsymbol|mathbf)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: GREEK.get(m.group(1), MATHSYM.get(m.group(1), "")), s)
    return re.sub(r"[{}]", "", s).strip()


def to_wordlatex(s: str) -> str:
    """LaTeX math -> the LaTeX subset that Word's equation editor builds up (Word set to LaTeX input; build_docx.ps1
    checks this). Only constructs Word does not accept are rewritten."""
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\operatorname\{([^}]*)\}", r"\\mathrm{\1}", s)
    while "\\ensuremath{" in s:  # unwrap (balanced braces: the argument may contain groups such as 10^{-5})
        k = s.index("\\ensuremath{")
        inner, end = group(s, k + len("\\ensuremath"))
        s = s[:k] + "{" + inner + "}" + s[end:]
    s = re.sub(r"\^\\circ\s*\\mathrm\{([A-Za-z])\}", r"^{\\circ}\\mathrm{\1}", s)
    s = re.sub(r"\^\{\\prime\\prime\\prime\}", "'''", s)
    s = re.sub(r"\^\{\\prime\\prime\}", "''", s)
    s = re.sub(r"\^\{\\prime\}", "'", s)
    s = s.replace("\\%", "%").replace("~", "\\ ").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # Word's LaTeX parser drops a bare symbol command after \, and lets an unbraced script swallow the next command:
    # brace every Greek letter / ordinary symbol and every single-token script argument
    sym = "|".join(sorted(list(GREEK) + ["partial", "nabla", "infty", "top"], key=len, reverse=True))
    s = re.sub(r"\\(" + sym + r")(?![A-Za-z])", r"{\\\1}", s)
    s = re.sub(r"([\^_])(\\[A-Za-z]+|[A-Za-z0-9*])", r"\1{\2}", s)
    s = re.sub(r"\\(boldsymbol|mathbf)\{([^{}]*)\}", r"{\\\1{\2}}", s)
    s = re.sub(r"(?<!\{)\\mathrm\{([^{}]*)\}", r"{\\mathrm{\1}}", s)
    return re.sub(r"\s+", " ", s).strip()


def _skip(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \n":
        i += 1
    return i


def arg(s: str, i: int) -> tuple[str, int]:
    """A command argument: a brace group, or a single token (one character or one \\command)."""
    i = _skip(s, i)
    if s[i] == "{":
        return group(s, i)
    m = re.match(r"\\[A-Za-z]+|.", s[i:])
    return m.group(0), i + len(m.group(0))


# ----------------------------------------------------------------------------------------------- inline text -> runs
class Ctx:
    def __init__(self, labels: dict, cites: list, bibkeys: set):
        self.labels, self.cites, self.bibkeys = labels, cites, bibkeys

    def cite(self, keys: list[str]) -> str:
        nums = []
        for k in keys:
            if k not in self.bibkeys:
                WARN.append(f"cite: {k} not in refs.bib")
            if k not in self.cites:
                self.cites.append(k)
            nums.append(self.cites.index(k) + 1)
        nums = sorted(set(nums))
        parts, j = [], 0
        while j < len(nums):
            k = j
            while k + 1 < len(nums) and nums[k + 1] == nums[k] + 1:
                k += 1
            parts.append(str(nums[j]) if k == j else (f"{nums[j]},{nums[k]}" if k == j + 1 else f"{nums[j]}–{nums[k]}"))
            j = k + 1
        return "[" + ",".join(parts) + "]"


def runs(s: str, ctx: Ctx, style: dict | None = None) -> list[dict]:
    style = style or {}
    out: list[dict] = []
    buf: list[str] = []

    def flush():
        if buf:
            out.append({"k": "t", "s": "".join(buf), **style})
            buf.clear()

    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "$":
            j = s.index("$", i + 1)
            inner = s[i + 1 : j].strip()
            scr = re.fullmatch(r"([\^_])\{?([^{}$]*)\}?", inner)
            if inner in ("^\\circ", "^{\\circ}"):  # 80\,$^\circ$C -> 80 °C
                buf.append("°")
            elif inner == "-":  # a lone $-$: the minus character
                buf.append("−")
            elif scr:  # a bare unit exponent such as m$^{-3}$: superscript / subscript text, not an equation
                flush()
                out.append({"k": "t", "s": scr.group(2).replace("-", "−"), **style,
                            ("sup" if scr.group(1) == "^" else "sub"): True})
            else:
                flush()
                out.append({"k": "m", "s": to_wordlatex(inner)})
            i = j + 1
        elif ch == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            name = m.group(1)
            i += len(m.group(0))
            if name in ("emph", "textit"):
                a, i = arg(s, i)
                flush()
                out += runs(a, ctx, {**style, "i": True})
            elif name == "textbf":
                a, i = arg(s, i)
                flush()
                out += runs(a, ctx, {**style, "b": True})
            elif name == "texttt":
                a, i = arg(s, i)
                flush()
                out += runs(a, ctx, {**style, "tt": True})
            elif name in ("cite", "citep", "citet"):
                a, i = arg(s, i)
                buf.append(ctx.cite([k.strip() for k in a.split(",")]))
            elif name in ("ref", "eqref"):
                a, i = arg(s, i)
                n = ctx.labels.get(a)
                if n is None:
                    WARN.append(f"ref: unknown label {a}")
                    n = "??"
                buf.append(f"({n})" if name == "eqref" else str(n))
            elif name == "ensuremath":
                a, i = arg(s, i)
                sci = re.fullmatch(r"\s*(\\ensuremath\{-\}|-)?([\d.]+)\\times\s*10\^\{(-?\d+)\}\s*", a)
                if a.strip() == "-":  # the generated minus sign: the character, not an equation
                    buf.append("−")
                elif sci:  # a plain number a x 10^b: text with a superscript exponent, not an equation
                    buf.append(("−" if sci.group(1) else "") + sci.group(2) + " × 10")
                    flush()
                    out.append({"k": "t", "s": sci.group(3).replace("-", "−"), **style, "sup": True})
                else:
                    flush()
                    out.append({"k": "m", "s": to_wordlatex(a)})
            elif name == "url":
                a, i = arg(s, i)
                buf.append(a)
            elif name in ACCENT and i < len(s):
                if s[i] == "{":
                    a, i = group(s, i)
                else:
                    a, i = s[i], i + 1
                if name == "c":
                    a = a.strip()
                buf.append(a + ACCENT[name])
            elif name in TEXTSYM:
                buf.append(TEXTSYM[name])
            elif name == "\\":
                buf.append("\n")
            elif name in ("centering", "small", "scriptsize", "footnotesize", "normalsize", "noindent", "label"):
                if name == "label":
                    _, i = arg(s, i)
            elif name == "item":
                buf.append("")
            else:
                WARN.append(f"text: unknown \\{name}")
        elif ch == "~":
            buf.append("\u00a0")
            i += 1
        elif s.startswith("---", i):
            buf.append("—")
            i += 3
        elif s.startswith("--", i):
            buf.append("–")
            i += 2
        elif s.startswith("``", i):
            buf.append("“")
            i += 2
        elif s.startswith("''", i):
            buf.append("”")
            i += 2
        elif ch == "`":
            buf.append("‘")
            i += 1
        elif ch == "'":
            buf.append("’")
            i += 1
        elif ch in "{}":
            i += 1
        elif ch == "\n":
            buf.append(" ")
            i += 1
        else:
            buf.append(ch)
            i += 1
    flush()
    # collapse whitespace inside text runs; keep a single space at run joins; composed characters (é, not e + ´)
    for r in out:
        if r["k"] == "t":
            r["s"] = unicodedata.normalize("NFC", re.sub(r"[ \t]+", " ", r["s"]))
    if out and out[0]["k"] == "t":
        out[0]["s"] = out[0]["s"].lstrip()
    if out and out[-1]["k"] == "t":
        out[-1]["s"] = out[-1]["s"].rstrip()
    return [r for r in out if r["k"] == "m" or r["s"]]


# ----------------------------------------------------------------------------------------------- bibliography
def bib(path: Path) -> dict[str, dict]:
    s = path.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),", s):
        j = s.index("{", m.start())
        body, _ = group(s, j)
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{", body):
            v, _ = group(body, f.end() - 1)
            fields[f.group(1).lower()] = v
        out[m.group(2).strip()] = {"type": m.group(1).lower(), **fields}
    return out


def bib_runs(e: dict, ctx: Ctx) -> list[dict]:
    def clean(x):
        return re.sub(r"\s+", " ", x).strip()

    names = [clean(a) for a in e.get("author", "").split(" and ")]
    shown = []
    for n in names:
        if n == "others":
            shown.append("et al.")
            continue
        if "," in n:
            last, first = [x.strip() for x in n.split(",", 1)]
            shown.append(f"{first} {last}")
        else:
            shown.append(n)
    if shown and shown[-1] == "et al.":
        auth = ", ".join(shown[:-1]) + " et al."
    else:
        auth = ", ".join(shown)
    parts = [auth, e.get("title", "")]
    if e["type"] == "article":
        jn = e.get("journal", "")
        vol = e.get("volume", "")
        num = f" ({e['number']})" if e.get("number") else ""
        pg = f" {e['pages']}" if e.get("pages") else ""
        parts.append(f"\\emph{{{jn}}} {vol}{num} ({e.get('year', '')}){pg}".strip())
    elif e["type"] == "inproceedings":
        vol = f" {e['volume']}" if e.get("volume") else ""
        pg = f", pp. {e['pages']}" if e.get("pages") else ""
        parts.append(f"in: \\emph{{{e.get('booktitle', '')}}}{vol} ({e.get('year', '')}){pg}")
    elif e["type"] == "book":
        parts.append(f"{e.get('publisher', '')}, {e.get('year', '')}")
    else:
        parts.append(f"{e.get('note', '')} ({e.get('year', '')})")
    if e.get("doi"):
        parts.append(f"doi:{e['doi']}")
    txt = ", ".join(p for p in parts if p) + "."
    return runs(txt, ctx)


# ----------------------------------------------------------------------------------------------- blocks
ENV = re.compile(r"\\begin\{(equation|figure|table|itemize|enumerate|nomenclature|keyterms)\*?\}")


def split_body(body: str) -> list[tuple[str, object]]:
    """Raw blocks in document order: ('sec', (level, star, title, label)), ('par', text), ('env', (name, content)),
    ('appendix', None), ('bib', None)."""
    items = []
    pos = 0
    tok = re.compile(r"\\(section|subsection|paragraph)(\*?)\{|"
                     r"\\begin\{(equation|figure|table|itemize|enumerate|nomenclature|keyterms)\}|"
                     r"\\appendix\b|\\bibliography\{")
    while True:
        m = tok.search(body, pos)
        text = body[pos : m.start() if m else len(body)]
        for para in re.split(r"\n\s*\n", text):
            if para.strip():
                items.append(("par", para.strip()))
        if not m:
            break
        if m.group(1):
            title, j = group(body, m.end() - 1)
            lab = None
            lm = re.match(r"\s*\\label\{([^}]+)\}", body[j:])
            if lm:
                lab, j = lm.group(1), j + lm.end()
            items.append(("sec", (m.group(1), bool(m.group(2)), title, lab)))
            pos = j
        elif m.group(3):
            name = m.group(3)
            end = body.index(f"\\end{{{name}}}", m.end())
            items.append(("env", (name, body[m.end() : end])))
            pos = end + len(f"\\end{{{name}}}")
        elif m.group(0).startswith("\\appendix"):
            items.append(("appendix", None))
            pos = m.end()
        else:
            _, j = group(body, m.end() - 1)
            items.append(("bib", None))
            pos = j
    return items


def number(items: list) -> dict:
    labels: dict[str, str] = {}
    sec = sub = eq = fig = tab = 0
    app = False
    for kind, val in items:
        if kind == "appendix":
            app, sec = True, 0
        elif kind == "sec":
            lvl, star, _, lab = val
            if star or lvl == "paragraph":
                continue
            if lvl == "section":
                sec, sub = sec + 1, 0
            else:
                sub += 1
            n = (chr(64 + sec) if app else str(sec)) + (f".{sub}" if lvl == "subsection" else "")
            if lab:
                labels[lab] = n
        elif kind == "env":
            name, content = val
            lab = re.search(r"\\label\{([^}]+)\}", content)
            if name == "equation":
                eq += 1
                if lab:
                    labels[lab.group(1)] = str(eq)
            elif name == "figure":
                fig += 1
                if lab:
                    labels[lab.group(1)] = str(fig)
            elif name == "table":
                tab += 1
                if lab:
                    labels[lab.group(1)] = str(tab)
    return labels


def table_rows(content: str, ctx: Ctx) -> tuple[list, int, list]:
    """Rows of cell runs, the number of header rows, and the 1-based rows that are one merged cell (\\multicolumn
    spanning the table: group headings)."""
    tb = re.search(r"\\begin\{tabular\}\{", content)
    spec, j = group(content, tb.end() - 1)
    end = content.index("\\end{tabular}", j)
    body = content[j:end]
    ncol = len(re.findall(r"[lcrp](?:\{[^}]*\})?", spec))
    head = 0
    rows, merged = [], []
    for chunk in re.split(r"\\\\", body):
        if "\\midrule" in chunk and rows and head == 0:
            head = len(rows)
        line = re.sub(r"\\(toprule|midrule|bottomrule)", "", chunk).strip()
        if not line:
            continue
        mc = re.fullmatch(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}", line, flags=re.S)
        if mc:
            rows.append([runs(mc.group(2).strip(), ctx)] + [[]] * (ncol - 1))
            if int(mc.group(1)) >= ncol:
                merged.append(len(rows))
            continue
        rows.append([runs(c.strip(), ctx) for c in re.split(r"(?<!\\)&", line)])
    rows = [r + [[]] * (ncol - len(r)) for r in rows]
    return rows, head, merged


def definitions(content: str, ctx: Ctx) -> list[dict]:
    """Items of a nomenclature / keyterms list: \\item[term] definition; an item without a definition is a group
    heading. The term is the balanced [...] group after \\item."""
    out = []
    for raw in re.split(r"\\item\b", content)[1:]:
        raw = raw.strip()
        term, rest = "", raw
        if raw.startswith("["):
            depth = 0
            for j, ch in enumerate(raw):
                depth += ch == "["
                depth -= ch == "]"
                if depth == 0:
                    term, rest = raw[1:j], raw[j + 1 :]
                    break
        rest = rest.strip()
        out.append({"term": runs(term, ctx), "def": runs(rest, ctx), "group": not rest})
    return out


def build(paper: Path) -> dict:
    mac = macros(paper)
    src = inline_inputs(read(paper / "main.tex"), paper)
    src = re.sub(r"\\input\{numbers\}", "", src)
    src = expand(src, mac)
    bibd = bib(paper / "refs.bib")
    doc: dict = {"blocks": []}
    tm = re.search(r"\\title\{", src)
    doc["title"], _ = group(src, tm.end() - 1)
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1)
    kw = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", src, re.S).group(1)
    hl = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", src, re.S)
    body = src[src.index("\\end{frontmatter}") + len("\\end{frontmatter}") : src.index("\\end{document}")]
    body = re.sub(r"\\bibliographystyle\{[^}]*\}", "", body)
    items = split_body(body)
    labels = number(items)
    ctx = Ctx(labels, [], set(bibd))
    doc["title"] = "".join(r.get("s", "") for r in runs(doc["title"], ctx))
    doc["authors"] = "[Author Name] — [Affiliation] — [e-mail]"
    doc["abstract"] = runs(ab, ctx)
    doc["highlights"] = [runs(x.strip(), ctx) for x in re.split(r"\\item\b", hl.group(1))[1:]] if hl else []
    doc["running"] = "".join(r.get("s", "") for r in runs(mac.get("runningtitle", ""), ctx))
    doc["keywords"] = "; ".join(
        "".join(r.get("s", "") for r in runs(k.strip(), ctx)) for k in re.sub(r"\s+", " ", kw).split("\\sep")
    )
    B = doc["blocks"]
    sec = sub = eq = fig = tab = 0
    app = False
    pend_par = None
    for kind, val in items:
        if kind == "appendix":
            app, sec = True, 0
            B.append({"t": "pagebreak"})
        elif kind == "sec":
            lvl, star, title, _ = val
            tr = runs(title, ctx)
            ttxt = "".join(r["s"] if r["k"] == "t" else math_plain(r["s"]) for r in tr)
            if lvl == "paragraph":
                pend_par = ttxt
                continue
            if star:
                B.append({"t": "h1", "s": ttxt})
                continue
            if lvl == "section":
                sec, sub = sec + 1, 0
                n = ("Appendix " + chr(64 + sec) + "." if app else f"{sec}.")
                B.append({"t": "h1", "s": f"{n} {ttxt}"})
            else:
                sub += 1
                n = f"{chr(64 + sec) if app else sec}.{sub}."
                B.append({"t": "h2", "s": f"{n} {ttxt}"})
        elif kind == "par":
            r = runs(val, ctx)
            if pend_par:
                r = [{"k": "t", "s": pend_par + ". ", "b": True}] + r
                pend_par = None
            if r:
                B.append({"t": "p", "runs": r})
        elif kind == "env":
            name, content = val
            if pend_par:  # a \paragraph title followed directly by an environment (e.g. a list): keep the title
                B.append({"t": "p", "runs": [{"k": "t", "s": pend_par + ".", "b": True}]})
                pend_par = None
            if name == "equation":
                eq += 1
                m = re.sub(r"\\label\{[^}]*\}", "", content).strip().rstrip(",.")
                tail = content.strip()[-1] if content.strip()[-1] in ",." else ""
                B.append({"t": "eq", "s": to_wordlatex(m) + tail, "num": f"({eq})"})
            elif name == "figure":
                fig += 1
                g = re.search(r"\\includegraphics(?:\[width=([0-9.]*)\\linewidth\])?\{([^}]+)\}", content)
                cap, _ = group(content, content.index("\\caption{") + len("\\caption"))
                B.append({
                    "t": "figure",
                    "path": str((paper / "figures" / (g.group(2) + ".png")).resolve()),
                    "width": float(g.group(1) or 1.0),
                    "caption": [{"k": "t", "s": f"Figure {fig}. ", "b": True, "acc": True}] + runs(cap, ctx),
                })
            elif name == "table":
                tab += 1
                cap, _ = group(content, content.index("\\caption{") + len("\\caption"))
                rows, head, merged = table_rows(content, ctx)
                B.append({
                    "t": "table",
                    "caption": [{"k": "t", "s": f"Table {tab}. ", "b": True, "acc": True}] + runs(cap, ctx),
                    "rows": rows,
                    "head": head,
                    "merged": merged,
                    "small": "\\scriptsize" in content or "\\small" in content,
                })
            elif name in ("nomenclature", "keyterms"):
                if name == "nomenclature":
                    B.append({"t": "h1", "s": "Nomenclature"})
                B.append({"t": "deftable", "kind": name, "rows": definitions(content, ctx)})
            else:
                parts = [x for x in re.split(r"\\item\b", content)[1:]]
                B.append({"t": "list", "ordered": name == "enumerate", "items": [runs(x.strip(), ctx) for x in parts]})
        elif kind == "bib":
            B.append({"t": "h1", "s": "References"})
            B.append({"t": "refs", "items": [bib_runs(bibd[k], ctx) for k in list(ctx.cites)]})
    unused = sorted(set(bibd) - set(ctx.cites))
    if unused:
        WARN.append(f"bib entries never cited: {unused}")
    doc["warnings"] = sorted(set(WARN))
    return doc


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    paper = (ROOT / (argv[0] if argv else "paper/paper2")).resolve()
    doc = build(paper)
    out = paper / "build" / "doc.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    n = {t: sum(1 for b in doc["blocks"] if b["t"] == t)
         for t in ("h1", "h2", "p", "eq", "figure", "table", "list", "deftable")}
    n["highlights"] = len(doc["highlights"])
    print(f"tex2json: {out.relative_to(ROOT)} {n}; warnings: {len(doc['warnings'])}")
    for w in doc["warnings"]:
        print("   ", w)
    return 1 if doc["warnings"] else 0


if __name__ == "__main__":
    sys.exit(main())
