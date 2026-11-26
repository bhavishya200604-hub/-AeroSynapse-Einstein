"""LaTeX -> Word document model (paper/tools/tex2json.py) and the backslash-less macro lint rule."""

from aerosynapse.thermal.paper_tools import lint_paper, tex2json

MAIN = r"""\documentclass{elsarticle}
\newcommand{\Rey}{\mathrm{Re}}
\input{numbers}
\begin{document}
\begin{frontmatter}
\title{A test paper}
\begin{abstract}Error \Err{} \cite{B}.\end{abstract}
\begin{keyword}one \sep Cram\'er--Rao\end{keyword}
\end{frontmatter}
\section{Introduction}\label{sec:intro}
See \cite{B,A} and Equation~\eqref{eq:one} in Section~\ref{sec:intro}; heat flux $10^5$\,W\,m$^{-2}$ at 80\,$^\circ$C;
change \Neg{}.
\begin{equation}
\Rey = \frac{UL}{\nu} \label{eq:one}
\end{equation}
\section{Conclusions}
Done.
\bibliographystyle{elsarticle-num}
\bibliography{refs}
\end{document}
"""
BIB = """@article{A,
  author = {Doe, J. and others},
  title = {First},
  journal = {J. Test},
  volume = {1},
  pages = {1--2},
  year = {2020}
}
@book{B,
  author = {Roe, R.},
  title = {Second},
  publisher = {Pub},
  year = {2021}
}
"""


def make(tmp_path):
    p = tmp_path / "paper"
    (p / "sections").mkdir(parents=True)
    (p / "main.tex").write_text(MAIN, encoding="utf-8")
    (p / "numbers.tex").write_text(
        "\\newcommand{\\Err}{4.7\\,\\%}\n\\newcommand{\\Neg}{\\ensuremath{-}3\\,\\%}\n", encoding="utf-8"
    )
    (p / "refs.bib").write_text(BIB, encoding="utf-8")
    return p


def text(runs):
    return "".join(r.get("s", "") for r in runs)


def test_document_model(tmp_path):
    doc = tex2json.build(make(tmp_path))
    assert doc["warnings"] == []
    assert doc["keywords"] == "one; Cramér–Rao"
    assert text(doc["abstract"]) == "Error 4.7 % [1]."  # B is cited first -> [1]
    p = next(b for b in doc["blocks"] if b["t"] == "p")
    t = text(p["runs"])
    assert "[1,2]" in t and "(1)" in t and "Section 1" in t
    assert "80 °C" in t and "−3 %" in t  # degree sign and true minus, not equations
    assert any(r.get("sup") and r["s"] == "−2" for r in p["runs"])  # m^{-2} as superscript text
    eq = next(b for b in doc["blocks"] if b["t"] == "eq")
    assert eq["num"] == "(1)" and "\\frac" in eq["s"] and "{\\nu}" in eq["s"] and "{\\mathrm{Re}}" in eq["s"]
    refs = next(b for b in doc["blocks"] if b["t"] == "refs")["items"]
    assert text(refs[0]).startswith("R. Roe, Second, Pub, 2021") and "J. Doe et al." in text(refs[1])


def test_lint_flags_macro_without_backslash(tmp_path):
    p = make(tmp_path)
    (p / "main.tex").write_text(MAIN.replace("Done.", "Done Err{}."), encoding="utf-8")
    bad, _ = lint_paper.check_macros(p, lint_paper.read_tex(p))
    assert any("Err{} without backslash" in b for b in bad)


def test_nested_ensuremath_in_math_and_paragraph_before_list(tmp_path):
    p = make(tmp_path)
    body = r"""\paragraph{Limitations}
\begin{itemize}
\item one
\end{itemize}
Small $p = \Sci{}$ here.
"""
    (p / "main.tex").write_text(MAIN.replace("Done.", body), encoding="utf-8")
    (p / "numbers.tex").write_text(
        r"""\newcommand{\Err}{4.7\,\%}
\newcommand{\Neg}{\ensuremath{-}3\,\%}
\newcommand{\Sci}{\ensuremath{1.4\times 10^{-5}}}
""",
        encoding="utf-8",
    )
    doc = tex2json.build(p)
    blocks = doc["blocks"]
    i = next(k for k, b in enumerate(blocks) if b["t"] == "list")
    assert blocks[i - 1]["t"] == "p" and text(blocks[i - 1]["runs"]) == "Limitations."
    m = [r["s"] for b in blocks if b["t"] == "p" for r in b["runs"] if r["k"] == "m"]
    assert any(r"1.4\times 10^{-5}" in s and "ensuremath" not in s for s in m)


HIGHLIGHTS = r"""\begin{highlights}
\item First point \Err{}.
\item Second point.
\end{highlights}
\begin{abstract}"""
NOMEN = r"""\begin{nomenclature}
\item[\textit{Greek symbols}]
\item[$\theta$] temperature
\end{nomenclature}
\section{Introduction}"""
GLOSS = r"""Done.
\begin{keyterms}
\item[Coverage] fraction of [covered] points
\end{keyterms}
"""


def test_highlights_running_title_and_definition_tables(tmp_path):
    p = make(tmp_path)
    s = MAIN.replace(r"\begin{abstract}", HIGHLIGHTS)
    s = s.replace(r"\input{numbers}", "\\input{numbers}\n\\newcommand{\\runningtitle}{Short title}")
    s = s.replace(r"\section{Introduction}", NOMEN, 1).replace("Done.", GLOSS)
    (p / "main.tex").write_text(s, encoding="utf-8")
    doc = tex2json.build(p)
    assert doc["warnings"] == []
    assert [text(h) for h in doc["highlights"]] == ["First point 4.7 %.", "Second point."]
    assert doc["running"] == "Short title"
    blocks = doc["blocks"]
    i = next(k for k, b in enumerate(blocks) if b["t"] == "deftable")
    assert blocks[i - 1] == {"t": "h1", "s": "Nomenclature"} and blocks[i]["kind"] == "nomenclature"
    rows = blocks[i]["rows"]
    assert rows[0]["group"] and text(rows[0]["term"]) == "Greek symbols"
    assert not rows[1]["group"] and rows[1]["term"][0]["k"] == "m" and text(rows[1]["def"]) == "temperature"
    g = [b for b in blocks if b["t"] == "deftable" and b["kind"] == "keyterms"][0]["rows"][0]
    assert text(g["term"]) == "Coverage" and text(g["def"]) == "fraction of [covered] points"  # brackets kept
    assert {"t": "h1", "s": "1. Introduction"} in blocks  # the nomenclature does not shift section numbers
