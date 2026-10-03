from __future__ import annotations


def format_params(n: int) -> str:
    if n >= 1_000_000_000_000:
        return f"{n / 1e12:.2f}T"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def format_number(n: int) -> str:
    return f"{n:,}"


def shorten(name: str, max_len: int = 44) -> str:
    if len(name) <= max_len:
        return name
    return "..." + name[-(max_len - 3):]