"""Read files written by the original repository, whose modules were called ``src.*`` and ``experiments.*``."""

from __future__ import annotations

import io
import pickle
from typing import Any


def modname(name: str) -> str:
    if name.startswith("src."):
        return "aerosynapse.thermal." + name[4:]
    if name.startswith("experiments."):
        return "aerosynapse.thermal.experiments." + name[len("experiments.") :]
    if name.startswith("paper.tools"):
        return "aerosynapse.thermal.paper_tools" + name[len("paper.tools") :]
    return name


class Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(modname(module), name)


def loads(blob: bytes) -> Any:
    return Unpickler(io.BytesIO(blob)).load()
