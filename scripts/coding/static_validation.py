#!/usr/bin/env python3
"""AST-based static validation of generated analysis code, before it ever runs."""

from __future__ import annotations

import ast
from typing import Any

ALLOWED_IMPORT_ROOTS = {
    "json",
    "math",
    "pathlib",
    "sys",
    "warnings",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
    "linearmodels",
    "networkx",
}
FORBIDDEN_SNIPPETS = [
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "subprocess",
    "requests",
    "socket",
    "urllib",
    "http",
    "os.",
    "shutil",
    "pickle",
]


def validate_analysis_code(code: str) -> dict[str, Any]:
    issues: list[str] = []
    if not code.strip():
        return {"valid": False, "issues": ["empty generated code"]}
    lowered = code.lower()
    for snippet in FORBIDDEN_SNIPPETS:
        if snippet in lowered:
            issues.append(f"forbidden code snippet: {snippet}")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"valid": False, "issues": [f"syntax error: {exc}"]}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    issues.append(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                issues.append(f"disallowed import: {node.module}")
    return {"valid": not issues, "issues": issues}
