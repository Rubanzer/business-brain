"""Tests for analysis/tools/llm_gateway.py — LLM abstraction layer."""

import json

import pytest

from business_brain.analysis.tools.llm_gateway import (
    _cache_key,
    _extract_json,
)


# ---------------------------------------------------------------------------
# _extract_json (robust JSON extraction from LLM output)
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self):
        raw = '{"key": "value", "num": 42}'
        result = _extract_json(raw)
        assert result == {"key": "value", "num": 42}

    def test_json_in_code_block(self):
        raw = 'Here is the result:\n```json\n{"category": "A", "score": 0.9}\n```\nDone.'
        result = _extract_json(raw)
        assert result["category"] == "A"
        assert result["score"] == 0.9

    def test_json_in_plain_code_block(self):
        raw = '```\n{"x": 1}\n```'
        result = _extract_json(raw)
        assert result == {"x": 1}

    def test_json_embedded_in_text(self):
        raw = 'The analysis shows {"result": "good", "confidence": 0.85} which means it passed.'
        result = _extract_json(raw)
        assert result["result"] == "good"

    def test_no_json_returns_none(self):
        raw = "This is just plain text with no JSON."
        result = _extract_json(raw)
        assert result is None

    def test_invalid_json_returns_none(self):
        raw = '{key: value}'
        result = _extract_json(raw)
        assert result is None

    def test_nested_json(self):
        raw = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = _extract_json(raw)
        assert result["outer"]["inner"] == [1, 2, 3]
        assert result["flag"] is True

    def test_json_with_whitespace(self):
        raw = '   \n  {"key": "value"}\n  '
        result = _extract_json(raw)
        assert result["key"] == "value"

    def test_multiple_code_blocks_picks_first_valid(self):
        raw = '```\nnot json\n```\n```json\n{"valid": true}\n```'
        result = _extract_json(raw)
        assert result["valid"] is True

    def test_json_with_braces_in_text(self):
        raw = 'Some text before {"answer": 42} and after'
        result = _extract_json(raw)
        assert result is not None
        assert result["answer"] == 42


# ---------------------------------------------------------------------------
# _cache_key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_deterministic(self):
        k1 = _cache_key("test prompt")
        k2 = _cache_key("test prompt")
        assert k1 == k2

    def test_different_prompts_different_keys(self):
        k1 = _cache_key("prompt A")
        k2 = _cache_key("prompt B")
        assert k1 != k2

    def test_returns_hex_string(self):
        key = _cache_key("test")
        assert len(key) == 32  # md5 hex digest length
        assert all(c in "0123456789abcdef" for c in key)
