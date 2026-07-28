"""Unit tests for AST-based static validation of generated code."""
from __future__ import annotations

from scripts.coding.static_validation import validate_analysis_code


def test_empty_code_is_invalid() -> None:
    result = validate_analysis_code("")
    assert result["valid"] is False
    assert "empty generated code" in result["issues"]


def test_valid_code_with_allowed_imports() -> None:
    code = "import json\nimport pandas as pd\nprint(json.dumps({'status': 'ok'}))\n"
    result = validate_analysis_code(code)
    assert result["valid"] is True
    assert result["issues"] == []


def test_disallowed_import_is_flagged() -> None:
    code = "import requests\nprint('hi')\n"
    result = validate_analysis_code(code)
    assert result["valid"] is False
    assert any("requests" in issue for issue in result["issues"])


def test_forbidden_snippet_subprocess_is_flagged() -> None:
    code = "import json\nsubprocess.run(['ls'])\n"
    result = validate_analysis_code(code)
    assert result["valid"] is False
    assert any("subprocess" in issue for issue in result["issues"])


def test_forbidden_snippet_eval_is_flagged() -> None:
    code = "x = eval('1+1')\n"
    result = validate_analysis_code(code)
    assert result["valid"] is False
    assert any("eval(" in issue for issue in result["issues"])


def test_syntax_error_is_flagged_distinctly() -> None:
    code = "def f(:\n  pass"
    result = validate_analysis_code(code)
    assert result["valid"] is False
    assert any("syntax error" in issue for issue in result["issues"])


def test_from_import_of_disallowed_module_is_flagged() -> None:
    code = "from os import path\nprint(path)\n"
    result = validate_analysis_code(code)
    assert result["valid"] is False
    assert any("disallowed import" in issue for issue in result["issues"])
