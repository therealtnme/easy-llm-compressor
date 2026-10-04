"""Dataset schemas -> calibration text.

Calibration data arrives in whatever shape its author happened to use: a lone
text column, an input/response pair, an instruction/input/output triple, a chat
transcript inside a ``messages`` or ``conversations`` column, or something
bespoke. This module recognises the common conventions, renders every row to a
plain string, and -- when it cannot recognise the layout -- asks the user how
the dataset is formatted instead of guessing silently.

Nothing here invents text: a row that renders to an empty string is dropped,
and a format that cannot be resolved raises ``FormatError``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "FormatError",
    "DatasetFormat",
    "detect_format",
    "single_column_format",
    "format_from_answer",
    "resolve_format",
    "render_value",
    "render_chat",
    "looks_like_chat",
]

# --------------------------------------------------------------------------- #
# Vocabulary of column names seen in the wild
# --------------------------------------------------------------------------- #

CHAT_COLUMNS: tuple[str, ...] = (
    "messages", "message", "conversation", "conversations", "chat",
    "dialogue", "dialog", "turns", "conversation_messages", "chat_history",
)

INSTRUCTION_KEYS: tuple[str, ...] = (
    "instruction", "instructions", "inst", "prompt", "prompts", "question",
    "questions", "query", "task", "problem", "user", "user_message",
    "human", "request", "utterance", "command", "instruction_text",
    "prompt_text", "question_text", "q", "goal", "objective", "context_question",
)

INPUT_KEYS: tuple[str, ...] = (
    "input", "inputs", "in", "context", "passage", "document", "doc", "docs",
    "source", "src", "article", "premise", "text_a", "input_text", "context_text",
    "source_text", "source_sentence", "question_context", "support",
)

RESPONSE_KEYS: tuple[str, ...] = (
    "output", "outputs", "out", "response", "responses", "answer", "answers",
    "completion", "completions", "target", "targets", "solution", "solutions",
    "assistant", "gpt", "reply", "result", "expected", "reference", "label",
    "labels", "chosen", "correct", "output_text", "response_text", "answer_text",
    "completion_text", "summary", "a", "generated", "y",
)

TEXT_KEYS: tuple[str, ...] = (
    "text", "document", "document_text", "content", "contents", "body",
    "sentence", "paragraph", "passage", "article", "review", "abstract",
    "raw", "data", "description", "corpus", "sample", "line", "value",
    "utterance", "input", "in", "prompt", "instruction", "query", "question",
)

# columns that are almost certainly metadata rather than calibration text
NON_TEXT_HINTS: tuple[str, ...] = (
    "id", "ids", "idx", "index", "uid", "uuid", "key", "name", "title",
    "url", "link", "date", "timestamp", "time", "source_id", "doc_id",
    "document_id", "row_id", "split", "count", "length", "score", "label_id",
)

ROLE_ALIASES: dict[str, str] = {
    "human": "user", "user": "user", "question": "user", "ask": "user",
    "gpt": "assistant", "assistant": "assistant", "bot": "assistant",
    "ai": "assistant", "model": "assistant", "response": "assistant",
    "answer": "assistant", "system": "system", "prompt": "system",
    "instruction": "system", "tool": "tool", "function": "tool",
    "observation": "tool",
}

CONTENT_KEYS: tuple[str, ...] = (
    "content", "value", "text", "message", "utterance", "body", "response",
    "answer", "completion", "parts", "sentences", "message_text",
)

ROLE_KEYS: tuple[str, ...] = ("role", "from", "speaker", "author", "sender")

_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


class FormatError(ValueError):
    """The dataset layout could not be determined."""


# --------------------------------------------------------------------------- #
# Rendering values
# --------------------------------------------------------------------------- #

def _clean(text: str) -> str:
    """Collapse whitespace runs but keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(ln.split()) for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


def _maybe_json(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except (ValueError, TypeError):
        return value


def _turn_role(turn: Mapping[str, Any]) -> str:
    for key in ROLE_KEYS:
        raw = turn.get(key)
        if isinstance(raw, str) and raw.strip():
            role = raw.strip().lower()
            return ROLE_ALIASES.get(role, role)
    return ""


def _turn_content(turn: Mapping[str, Any]) -> str:
    for key in CONTENT_KEYS:
        raw = turn.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            if raw.strip():
                return raw
        elif isinstance(raw, (list, tuple, dict)):
            nested = render_value(raw)
            if nested.strip():
                return nested
        else:
            return str(raw)
    return ""


def looks_like_chat(value: Any) -> bool:
    """True when ``value`` is a transcript: a list of message dicts/strings."""
    if isinstance(value, str):
        value = _maybe_json(value)
    if isinstance(value, Mapping):
        for key in CHAT_COLUMNS:
            if key in value:
                return looks_like_chat(value[key])
        return False
    if not isinstance(value, (list, tuple)):
        return False
    if not value:
        return False
    head = value[0]
    if isinstance(head, Mapping):
        return bool(_turn_role(head) or _turn_content(head))
    return all(isinstance(item, str) for item in value)


def render_chat(value: Any) -> str:
    """Render a transcript (list of turns, ShareGPT-style dicts, or a blob)."""
    if isinstance(value, str):
        value = _maybe_json(value)
    if isinstance(value, Mapping):
        for key in CHAT_COLUMNS:
            if key in value:
                return render_chat(value[key])
        turn = _turn_content(value)
        return _clean(turn)
    if not isinstance(value, (list, tuple)):
        return _clean(str(value))
    lines: list[str] = []
    for turn in value:
        if isinstance(turn, Mapping):
            role = _turn_role(turn)
            content = _turn_content(turn)
            if not content:
                continue
            label = role.capitalize() if role else "User"
            lines.append(f"{label}: {content}")
        else:
            text = render_value(turn)
            if text:
                lines.append(text)
    return _clean("\n".join(lines))


def render_value(value: Any) -> str:
    """Render any dataset cell to a plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        if looks_like_chat(value):
            return render_chat(value)
        return _clean(value)
    if isinstance(value, Mapping):
        if looks_like_chat(value):
            return render_chat(value)
        return _clean(json.dumps(value, ensure_ascii=False))
    if isinstance(value, (list, tuple)):
        if looks_like_chat(value):
            return render_chat(value)
        return _clean("\n".join(render_value(v) for v in value))
    if isinstance(value, bool):
        return str(value)
    return str(value)


# --------------------------------------------------------------------------- #
# Format description
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DatasetFormat:
    """A resolved recipe for turning one dataset row into calibration text."""

    name: str
    kind: str                       # column | chat | template
    columns: tuple[str, ...] = ()
    template: Optional[str] = None
    confidence: str = "high"        # high | medium | low
    reason: str = ""

    def render(self, row: Mapping[str, Any]) -> str:
        if self.kind in ("column", "chat"):
            return render_value(row.get(self.columns[0])) if self.columns else ""
        return _render_template(self.template or "", row)

    def summary(self) -> str:
        parts = [f"format '{self.name}'", f"kind={self.kind}"]
        parts.append(f"columns={list(self.columns)}")
        if self.template:
            parts.append("template=" + " | ".join(self.template.split("\n\n")))
        parts.append(f"confidence={self.confidence}")
        if self.reason:
            parts.append(self.reason)
        return "; ".join(parts)


def _render_template(template: str, row: Mapping[str, Any]) -> str:
    lookup = {str(k).strip().lower(): k for k in row}

    def repl(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        actual = lookup.get(key.lower(), key)
        return render_value(row.get(actual)) if actual in row else ""

    filled = _PLACEHOLDER_RE.sub(repl, template)
    chunks = [_clean(c) for c in filled.split("\n\n")]
    return "\n\n".join(c for c in chunks if c)


def _template_from(columns: Sequence[str]) -> str:
    return "\n\n".join("{" + c + "}" for c in columns)


def single_column_format(column: str, confidence: str = "high",
                         reason: str = "") -> DatasetFormat:
    return DatasetFormat(name=f"column:{column}", kind="column",
                         columns=(column,), confidence=confidence,
                         reason=reason or f"single text column '{column}'")


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def _norm_map(columns: Iterable[str]) -> dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in columns}


def _first(norm: Mapping[str, str], keys: Sequence[str],
           exclude: Iterable[str] = ()) -> Optional[str]:
    skip = {str(x).strip().lower() for x in exclude if x}
    for key in keys:
        if key in norm and key not in skip:
            return norm[key]
    return None


def _string_columns(columns: Sequence[str], sample: Optional[Mapping[str, Any]],
                    extra_samples: Optional[Sequence[Mapping[str, Any]]] = None
                    ) -> list[str]:
    """Columns holding text. Uses the sample row(s) when values are available."""
    if sample is None:
        return [c for c in columns]
    rows = [r for r in ([sample] + list(extra_samples or [])) if isinstance(r, Mapping)]
    out: list[str] = []
    for column in columns:
        values = [r.get(column) for r in rows]
        if not values:
            out.append(column)
            continue
        if any(isinstance(v, (str, list, tuple, dict)) and v not in (None, "") for v in values):
            out.append(column)
    return out


def _looks_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, bool)


def _is_metadata(column: str) -> bool:
    key = column.strip().lower()
    if key in NON_TEXT_HINTS:
        return True
    return key.endswith(("_id", "_idx", "_index", "_url", "_link"))


def detect_format(columns: Sequence[str],
                  sample: Optional[Mapping[str, Any]] = None,
                  extra_samples: Optional[Sequence[Mapping[str, Any]]] = None
                  ) -> Optional[DatasetFormat]:
    """Best-effort recognition of a dataset schema. ``None`` -> ask the user."""
    columns = [str(c) for c in columns]
    norm = _norm_map(columns)
    samples = [r for r in ([sample] + list(extra_samples or []))
               if isinstance(r, Mapping)]

    # 1. chat transcripts stored in one column
    chat_hits = [norm[k] for k in CHAT_COLUMNS if k in norm]
    if chat_hits:
        for column in chat_hits:
            if sample is None or any(column in r for r in samples):
                return DatasetFormat(
                    name=f"chat:{column}", kind="chat", columns=(column,),
                    confidence="high",
                    reason=f"column '{column}' holds chat turns")

    # 2. prompt/response style pairs and instruction/input/output triples
    instruction = _first(norm, INSTRUCTION_KEYS)
    context = _first(norm, INPUT_KEYS, exclude=(instruction,))
    response = _first(norm, RESPONSE_KEYS, exclude=(instruction, context))
    if response is not None and (instruction is not None or context is not None):
        parts = [c for c in (instruction, context, response) if c]
        name = "/".join(p.strip().lower() for p in parts)
        return DatasetFormat(name=name, kind="template", columns=tuple(parts),
                             template=_template_from(parts), confidence="high",
                             reason="recognised " + "/".join(parts))

    # 3. a single obvious text column (ignore metadata columns)
    candidates = [c for c in columns if not _is_metadata(c)]
    if len(candidates) == 1:
        return single_column_format(
            candidates[0], confidence="high",
            reason=f"'{candidates[0]}' is the only non-metadata column")
    for key in TEXT_KEYS:
        if key in norm and norm[key] in candidates:
            return single_column_format(
                norm[key], confidence="high",
                reason=f"'{norm[key]}' is a recognised text column")

    # 4. heuristic fallbacks: text columns only, ordered as the dataset has them
    strings = [c for c in candidates if c in _string_columns(candidates, sample,
                                                              extra_samples)]
    if sample is not None and len(strings) > 1:
        numeric = [c for c in strings if all(_looks_numeric(r.get(c)) for r in samples)]
        strings = [c for c in strings if c not in numeric]
    if len(strings) == 1:
        return single_column_format(
            strings[0], confidence="medium",
            reason=f"'{strings[0]}' is the only text-looking column")
    if len(strings) == 2:
        head, tail = strings
        return DatasetFormat(name=f"pair:{head}/{tail}", kind="template",
                             columns=(head, tail),
                             template=_template_from((head, tail)),
                             confidence="low",
                             reason=(f"guessed that '{head}' is the input and "
                                     f"'{tail}' the response"))
    if len(strings) > 2:
        return DatasetFormat(name="concat", kind="template",
                             columns=tuple(strings),
                             template=_template_from(strings), confidence="low",
                             reason="concatenated every text column in order")
    return None


# --------------------------------------------------------------------------- #
# User-supplied formats
# --------------------------------------------------------------------------- #

def _alias_guess(key: str, columns: Sequence[str]) -> Optional[str]:
    """Map a role word ('output', 'prompt', ...) onto one of the columns."""
    norm = _norm_map(columns)
    low = key.strip().lower()
    instruction = _first(norm, INSTRUCTION_KEYS)
    context = _first(norm, INPUT_KEYS, exclude=(instruction,))
    response = _first(norm, RESPONSE_KEYS, exclude=(instruction, context))
    if low in INSTRUCTION_KEYS:
        return instruction or context or response
    if low in INPUT_KEYS:
        return context or instruction or response
    if low in RESPONSE_KEYS:
        return response or instruction or context
    fuzzy = [str(c) for c in columns
             if low and (low in str(c).strip().lower()
                         or str(c).strip().lower() in low)]
    return fuzzy[0] if len(fuzzy) == 1 else None


def _resolve_columns(tokens: Sequence[str], columns: Sequence[str]) -> tuple[str, ...]:
    norm = _norm_map(columns)
    resolved: list[str] = []
    for token in tokens:
        key = token.strip()
        if not key:
            continue
        actual = norm.get(key.lower())
        if actual is None:
            actual = _alias_guess(key, columns)
        if actual is None:
            raise FormatError(
                f"'{key}' is not a column of this dataset; available columns: "
                f"{list(columns)}")
        resolved.append(actual)
    if not resolved:
        raise FormatError("no columns given")
    return tuple(resolved)


def format_from_answer(answer: str, columns: Sequence[str],
                       sample: Optional[Mapping[str, Any]] = None
                       ) -> DatasetFormat:
    """Turn a user's description into a DatasetFormat.

    Accepts a column name, a comma/space separated column list, a chat column
    name, or a template with ``{column}`` placeholders.
    """
    text = (answer or "").strip()
    if not text:
        raise FormatError("empty answer")
    lowered = text.lower()

    chat_hits = [c for c in columns if c.strip().lower() in CHAT_COLUMNS]
    if lowered in ("chat", "messages", "conversation", "conversations", "transcript"):
        chat_hits = chat_hits or [c for c in columns
                                  if c.strip().lower() in CHAT_COLUMNS]
        if not chat_hits:
            raise FormatError(
                "no chat column found; name the column that holds the turns, "
                f"e.g. one of {list(columns)}")
        return DatasetFormat(name=f"chat:{chat_hits[0]}", kind="chat",
                             columns=(chat_hits[0],), confidence="user",
                             reason="chosen by the user")

    if "{" in text:
        placeholders = [m.group(1).strip() for m in _PLACEHOLDER_RE.finditer(text)]
        resolved = _resolve_columns(placeholders, columns)
        return DatasetFormat(name="custom template", kind="template",
                             columns=resolved, template=text,
                             confidence="user", reason="template from the user")

    pieces = [p.strip() for p in text.split(",") if p.strip()]
    if len(pieces) < 2:
        pieces = [p for p in re.split(r"\s+", text) if p]
    if len(pieces) == 1:
        column = _resolve_columns(pieces, columns)[0]
        if column.strip().lower() in CHAT_COLUMNS or (
                sample is not None and looks_like_chat(sample.get(column))):
            return DatasetFormat(name=f"chat:{column}", kind="chat",
                                 columns=(column,), confidence="user",
                                 reason="chosen by the user")
        return single_column_format(column, confidence="user",
                                    reason="chosen by the user")
    resolved = _resolve_columns(pieces, columns)
    return DatasetFormat(name="custom template", kind="template",
                         columns=resolved, template=_template_from(resolved),
                         confidence="user",
                         reason="column order from the user")


# --------------------------------------------------------------------------- #
# Resolution (explicit -> detect -> ask)
# --------------------------------------------------------------------------- #

def _default_answer(guess: Optional[DatasetFormat], columns: Sequence[str]) -> str:
    if guess is not None and guess.template:
        return guess.template
    if guess is not None and guess.columns:
        return guess.columns[0]
    if len(columns) == 2:
        return _template_from((str(columns[0]), str(columns[1])))
    return str(columns[0]) if columns else ""


def _ask_user(columns: Sequence[str], sample: Optional[Mapping[str, Any]],
              guess: Optional[DatasetFormat],
              notify: Optional[Callable[[str], None]]) -> DatasetFormat:
    def say(line: str) -> None:
        if notify:
            notify(line)

    say("could not auto-detect the dataset format; please describe it")
    say(f"  columns: {list(columns)}")
    if sample is not None:
        preview = ", ".join(f"{k}={type(v).__name__}" for k, v in list(sample.items())[:8])
        say(f"  first row: {preview}")
    if guess is not None:
        say(f"  best guess: {guess.summary()}")

    try:
        import typer
    except ImportError:  # pragma: no cover - typer is a hard dependency of the CLI
        raise FormatError(
            "could not auto-detect the dataset format and no prompt is "
            f"available; pass --dataset-format with a template over {list(columns)}")

    default = _default_answer(guess, columns)
    while True:
        answer = typer.prompt(
            "how is this dataset formatted? (a template using {column} names, "
            "a column name, or a comma separated column order)",
            default=default)
        try:
            return format_from_answer(answer, columns, sample)
        except FormatError as exc:
            typer.secho(f"  {exc}", fg=typer.colors.YELLOW)


def resolve_format(columns: Sequence[str],
                   field: str = "auto",
                   format_spec: Optional[str] = None,
                   sample: Optional[Mapping[str, Any]] = None,
                   extra_samples: Optional[Sequence[Mapping[str, Any]]] = None,
                   interactive: Optional[bool] = None,
                   notify: Optional[Callable[[str], None]] = None,
                   ) -> DatasetFormat:
    """Decide how to read a dataset: explicit request, detection, or the user."""
    columns = [str(c) for c in columns]
    norm = _norm_map(columns)

    if format_spec:
        return format_from_answer(format_spec, columns, sample)

    if field and field.strip().lower() not in ("auto", "*"):
        actual = norm.get(field.strip().lower())
        if actual is None:
            raise FormatError(
                f"field '{field}' not in dataset columns {columns}")
        if sample is not None and looks_like_chat(sample.get(actual)):
            return DatasetFormat(name=f"chat:{actual}", kind="chat",
                                 columns=(actual,), confidence="user",
                                 reason=f"'{actual}' holds chat turns")
        return single_column_format(actual, confidence="user",
                                    reason="requested via --dataset-field")

    detected = detect_format(columns, sample, extra_samples)
    if detected is not None and detected.confidence in ("high", "medium"):
        return detected

    if interactive is None:
        import sys
        try:
            interactive = bool(sys.stdin and sys.stdin.isatty())
        except Exception:
            interactive = False
    if interactive:
        return _ask_user(columns, sample, detected, notify)

    if detected is not None:  # low confidence, nobody to ask: use it, but say so
        if notify:
            notify(f"warning: {detected.reason}; assuming {detected.summary()}. "
                   "Pass --dataset-format to correct this.")
        return detected

    raise FormatError(
        "could not auto-detect the dataset format from columns "
        f"{columns}; pass --dataset-format with a template using {{column}} "
        "names (or --dataset-field to name the text column)")
