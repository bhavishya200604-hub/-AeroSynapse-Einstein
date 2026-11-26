"""Cross-platform task runner (Windows / Linux / macOS): ``nox -s lint typecheck tests``."""

import nox

nox.options.sessions = ["lint", "typecheck", "tests"]


@nox.session(python=False)
def lint(session: nox.Session) -> None:
    session.run("ruff", "check", "src", "tests")
    session.run("ruff", "format", "--check", "src", "tests")


@nox.session(python=False)
def typecheck(session: nox.Session) -> None:
    session.run("mypy")


@nox.session(python=False)
def tests(session: nox.Session) -> None:
    session.run("pytest", "-m", "not slow and not gpu", "--cov=aerosynapse", "--cov-report=term")


@nox.session(python=False)
def reproduce(session: nox.Session) -> None:
    session.run("python", "scripts/reproduce_all.py", "--profile", "quick")


@nox.session(python=False)
def docs(session: nox.Session) -> None:
    session.run("mkdocs", "build", "--strict")
