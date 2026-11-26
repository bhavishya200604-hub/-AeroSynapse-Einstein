"""Plugin registry: physics modules, solvers, models and optimisers register by name.

New components are added with a decorator and need no edit to core code::

    from aerosynapse.registry import SOLVERS

    @SOLVERS.register("my_solver")
    def my_solver(...): ...
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A named mapping from string keys to factories."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                raise KeyError(f"{self.kind} {name!r} already registered")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {sorted(self._items)}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))


PHYSICS: Registry[Any] = Registry("physics module")
SOLVERS: Registry[Any] = Registry("solver")
MODELS: Registry[Any] = Registry("model")
OPTIMIZERS: Registry[Any] = Registry("optimizer")
