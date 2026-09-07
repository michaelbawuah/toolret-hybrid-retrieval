from __future__ import annotations

import json
import re
from typing import Any


_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer used by BM25.
    """
    return [m.group(0).lower() for m in _TOKEN.finditer(text or "")]


def _parse_json_if_possible(value: Any) -> Any:
    """
    ToolRet corpus fields may contain JSON serialized as strings.
    If a string contains valid JSON, decode it.
    Otherwise return it unchanged.
    """
    if not isinstance(value, str):
        return value

    value = value.strip()

    if not value:
        return value

    if value[0] not in "{[":
        return value

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _collect_text(value: Any) -> list[str]:
    """
    Recursively extract useful textual information from nested
    dictionaries, lists, tuples, sets, and JSON strings.
    """
    value = _parse_json_if_possible(value)

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, (int, float, bool)):
        return [str(value)]

    if isinstance(value, dict):
        parts: list[str] = []

        # Prioritize semantically useful tool fields.
        priority_keys = (
            "name",
            "description",
            "parameters",
            "arguments",
            "doc_arguments",
        )

        processed = set()

        for key in priority_keys:
            if key in value:
                processed.add(key)

                # Include parameter/field names because they often
                # contain useful retrieval information.
                if key not in {"parameters", "arguments", "doc_arguments"}:
                    parts.append(str(key).replace("_", " "))

                parts.extend(_collect_text(value[key]))

        # Preserve any additional useful fields that may appear
        # in other ToolRet subsets.
        for key, item in value.items():
            if key in processed:
                continue

            # IDs themselves are generally not semantically useful.
            if key.lower() in {"id", "tool_id", "query_id"}:
                continue

            parts.append(str(key).replace("_", " "))
            parts.extend(_collect_text(item))

        return parts

    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []

        for item in value:
            parts.extend(_collect_text(item))

        return parts

    return [str(value)]


def flatten_tool(tool: Any) -> str:
    """
    Convert a ToolRet corpus document into searchable natural-language text.

    Handles:
    - normal Python dictionaries
    - nested dictionaries
    - parameter schemas
    - lists
    - JSON serialized inside strings
    - heterogeneous ToolRet subsets
    """
    parts = _collect_text(tool)

    cleaned_parts = []

    for part in parts:
        part = str(part).replace("_", " ")
        part = " ".join(part.split())

        if part:
            cleaned_parts.append(part)

    return " ".join(cleaned_parts)