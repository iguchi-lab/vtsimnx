"""名前で選ぶ部品カタログ。取り出すたびにコピーを返すので、その場で編集できる。"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeVar

T = TypeVar("T")


class PartCatalog(Mapping[str, T]):
    """登録した部品を名前で取り出す。callable なら呼び、それ以外は deepcopy する。"""

    def __init__(self, factories: Mapping[str, Callable[..., T] | T]) -> None:
        self._factories = dict(factories)

    def __getitem__(self, name: str) -> T:
        try:
            item = self._factories[name]
        except KeyError as exc:
            known = ", ".join(self._factories) or "(empty)"
            raise KeyError(f"unknown part {name!r}. available: {known}") from exc
        if callable(item):
            return item()
        return copy.deepcopy(item)

    def __iter__(self) -> Iterator[str]:
        return iter(self._factories)

    def __len__(self) -> int:
        return len(self._factories)

    def names(self) -> list[str]:
        return list(self._factories)


def snapshot(value: Any) -> Any:
    return copy.deepcopy(value)
