"""Render a tool document as text at one of three levels of detail.

The ablation in PROJECT.md section 8 asks what a retriever needs to see of a
tool. The same renderer is used when training and when evaluating, so a model
trained on names alone is also evaluated on names alone.

    full              the document as released (what the Stage 1 baselines saw)
    name_description  "name: description"
    name              the name alone
"""

import json

MODES = ("full", "name_description", "name")
NAME_KEYS = ("name", "api_call", "name_for_human")
DESCRIPTION_KEYS = ("description", "functionality", "description_for_human", "func_description")


def fields(document: str) -> tuple[str, str]:
    """(name, description) of a tool document, each empty if the document has none."""
    try:
        doc = json.loads(document)
    except ValueError:
        return "", ""
    if not isinstance(doc, dict):
        return "", ""
    name = next((str(doc[key]) for key in NAME_KEYS if doc.get(key)), "")
    description = next((str(doc[key]) for key in DESCRIPTION_KEYS if doc.get(key)), "")
    return name.strip(), description.strip()


def render_tool(document: str, mode: str = "full") -> str:
    if mode == "full":
        return document
    name, description = fields(document)
    if mode == "name":
        return name or description or document
    if mode == "name_description":
        return ": ".join(part for part in (name, description) if part) or document
    raise ValueError(f"unknown tool text mode: {mode}")
