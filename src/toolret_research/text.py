import re

_TOKEN = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN.finditer(text or "")]

def flatten_tool(tool: dict) -> str:
    parts = []
    if tool.get("name"):
        parts.append(str(tool["name"]).replace("_", " "))
    if tool.get("description"):
        parts.append(str(tool["description"]))

    params = tool.get("parameters") or {}
    if isinstance(params, dict):
        for name, spec in params.items():
            parts.append(str(name).replace("_", " "))
            if isinstance(spec, dict):
                if spec.get("description"):
                    parts.append(str(spec["description"]))
                if spec.get("type"):
                    parts.append(str(spec["type"]))
            elif spec is not None:
                parts.append(str(spec))
    return " ".join(parts)
