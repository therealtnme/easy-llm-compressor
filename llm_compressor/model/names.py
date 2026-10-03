"""Token matching helpers.

Substring matching on module names is a recurring source of silent bugs
(e.g. `"in"` matching `"intermediate"`, `"up"` matching `"output"`). All name
reasoning in this package goes through tokenisation + set intersection.
"""
from __future__ import annotations

import re

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def tokenize_name(name: str) -> set[str]:
    """Split a dotted/camelCase module name into lowercase tokens."""
    tokens: set[str] = set()
    for part in str(name).split("."):
        for chunk in _SPLIT.split(_CAMEL.sub(" ", part)):
            chunk = chunk.lower()
            if chunk:
                tokens.add(chunk)
    return tokens


def has_token(name: str, vocab: set[str]) -> bool:
    return bool(tokenize_name(name) & vocab)


def tail(name: str, depth: int = 1) -> str:
    parts = str(name).split(".")
    return ".".join(parts[-depth:])


def leaf(name: str) -> str:
    return str(name).rsplit(".", 1)[-1]


def common_prefix_depth(a: str, b: str) -> int:
    pa, pb = a.split("."), b.split(".")
    n = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        n += 1
    return n


def is_prefix(prefix: str, name: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")
