"""Ollama integration degrades to the heuristic planner on every failure mode."""

import httpx
import pytest

from scenepeek.core.config import get_settings
from scenepeek.ml import llm
from scenepeek.search.planner import plan


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    llm.reset_cache()
    monkeypatch.setattr(get_settings(), "ollama_enabled", False)
    monkeypatch.setattr(get_settings(), "ollama_auto", True)
    yield
    llm.reset_cache()


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_available_when_server_lists_model(monkeypatch):
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})

    monkeypatch.setattr(
        httpx, "get", lambda url, timeout: httpx.Client(transport=_mock_transport(handler)).get(url)
    )
    assert llm.is_available() is True
    assert llm.enabled() is True


def test_unavailable_when_model_not_pulled(monkeypatch):
    handler = lambda request: httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})  # noqa: E731
    monkeypatch.setattr(
        httpx, "get", lambda url, timeout: httpx.Client(transport=_mock_transport(handler)).get(url)
    )
    assert llm.is_available() is False


def test_unavailable_on_connection_error(monkeypatch):
    def boom(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", boom)
    assert llm.is_available() is False
    assert llm.enabled() is False


def test_availability_is_cached(monkeypatch):
    calls = {"n": 0}

    def get(url, timeout):
        calls["n"] += 1
        return httpx.Client(
            transport=_mock_transport(lambda r: httpx.Response(200, json={"models": []}))
        ).get(url)

    monkeypatch.setattr(httpx, "get", get)
    llm.is_available()
    llm.is_available()
    assert calls["n"] == 1


def test_decompose_malformed_json_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "_chat", lambda *a, **k: "not json at all")
    assert llm.decompose("q") is None


def test_decompose_timeout_falls_back(monkeypatch):
    def post(url, json, timeout):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx, "post", post)
    assert llm.decompose("q") is None


def test_planner_uses_llm_when_enabled(monkeypatch):
    monkeypatch.setattr(get_settings(), "ollama_enabled", True)
    monkeypatch.setattr(
        llm, "decompose", lambda q: {"speech": "hash tables", "visual": "a slide with a diagram", "ocr": ""}
    )
    p = plan("the part about hash tables when the slide shows a diagram")
    assert p.source == "llm"
    assert p.speech_q == "hash tables" and p.visual_q == "a slide with a diagram"
    assert "hash" in p.lexical_terms


def test_planner_falls_back_when_llm_fails(monkeypatch):
    monkeypatch.setattr(get_settings(), "ollama_enabled", True)
    monkeypatch.setattr(llm, "decompose", lambda q: None)
    p = plan("the part about hash tables")
    assert p.source == "heuristic"
