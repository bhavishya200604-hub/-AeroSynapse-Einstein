"""Paper tooling (Phase 6B): lint checks catch each defect class; Paper 2 passes; the Overleaf pack is complete."""

import zipfile

from aerosynapse.thermal.paper_tools import lint_paper, pack_overleaf

MAIN = r"""\documentclass{elsarticle}
\newcommand{\Rey}{\mathrm{Re}}
\input{numbers}
\begin{document}
\begin{frontmatter}
\begin{abstract}A study.\end{abstract}
\begin{keyword}x\end{keyword}
\end{frontmatter}
\input{sections/01}
\section*{Data availability} x
\section*{CRediT authorship contribution statement} x
\section*{Declaration of competing interest} x
\bibliography{refs}
\end{document}
"""
SEC_OK = r"""\section{Introduction}\label{sec:intro}
The error is \Err{} at $\Rey = 10$ \cite{Raissi2019} (Section~\ref{sec:intro}).
\section{Conclusions}
"""
BIB = "@article{Raissi2019,\n  title = {x}\n}\n"


def make(tmp_path, sec=SEC_OK, bib=BIB):
    (tmp_path / "sections").mkdir(parents=True)
    (tmp_path / "main.tex").write_text(MAIN, encoding="utf-8")
    (tmp_path / "numbers.tex").write_text("\\newcommand{\\Err}{1.2\\,\\%}% R0\n", encoding="utf-8")
    (tmp_path / "refs.bib").write_text(bib, encoding="utf-8")
    (tmp_path / "sections" / "01.tex").write_text(sec, encoding="utf-8")
    return tmp_path


def test_clean_paper_passes(tmp_path, capsys):
    assert lint_paper.main([str(make(tmp_path))]) == 0
    assert "lint_paper: PASS" in capsys.readouterr().out


def test_each_defect_is_caught(tmp_path, capsys):
    sec = SEC_OK + "\\Missing{} We optimize at 0.37 with a color map \\cite{Rahaman2019}. \\ref{fig:none}\n{"
    bib = BIB + "@article{Rahaman2019,\n  title = {y}\n}\n"
    assert lint_paper.main([str(make(tmp_path, sec, bib))]) == 1
    out = capsys.readouterr().out
    assert "undefined macro \\Missing" in out
    assert "not VERIFIED: Rahaman2019" in out  # UNVERIFIED in docs/LITERATURE.md
    assert "literal number(s) 0.37" in out
    assert "'optimize'" in out and "'color'" in out
    assert "unresolved reference fig:none" in out
    assert "unbalanced braces" in out


def test_claims_ok_marker_exempts_inputs(tmp_path):
    sec = SEC_OK + "The grid has 128 cells. % claims-ok: configuration input\n"
    assert lint_paper.main([str(make(tmp_path, sec))]) == 0


def test_paper2_passes_lint():
    assert lint_paper.main(["paper/paper2"]) == 0


def test_paper1_passes_lint():
    assert lint_paper.main(["paper/paper1"]) == 0


def test_overleaf_pack_contains_sources(tmp_path, monkeypatch):
    paper = make(tmp_path / "p")
    (paper / "figures").mkdir()
    (paper / "figures" / "fig_a.pdf").write_bytes(b"%PDF-1.4")
    (paper / "figures" / "fig_a.png").write_bytes(b"png")
    (paper / "tables").mkdir()
    (paper / "tables" / "tab.tex").write_text("1 & 2 \\\\", encoding="utf-8")
    monkeypatch.setattr(pack_overleaf, "ROOT", tmp_path)
    out = pack_overleaf.pack(paper)
    names = set(zipfile.ZipFile(out).namelist())
    assert {"main.tex", "numbers.tex", "refs.bib", "sections/01.tex", "tables/tab.tex", "figures/fig_a.pdf"} <= names
    assert "figures/fig_a.png" not in names
