from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import httpx
import pytest
from langchain_litellm import ChatLiteLLM

from taskboard_agent.config import ConfigError, _parse_agent_profile
from taskboard_agent.llm import LiteLLMClient, LLMError
from taskboard_agent.reasoning import reasoning_kwargs, validate_ollama_capabilities


@pytest.fixture
def endpoint():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if self.path == "/api/generate":
                result = {"model": "qwen3", "response": "ok", "done": True}
            elif self.path == "/api/chat":
                result = {"model": "qwen3", "message": {"role": "assistant", "content": "ok"}, "done": True}
            else:
                result = {"id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                          "model": "qwen3", "choices": [{"index": 0, "finish_reason": "stop",
                          "message": {"role": "assistant", "content": "ok"}}]}
            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("path", ["direct", "langchain"])
@pytest.mark.parametrize("backend,model,effort", [
    (None, "openai/qwen3", None),
    ("strata", "openai/qwen3", "none"),
    ("strata", "openai/qwen3", "low"),
    ("ollama", "openai/qwen3", "none"),
    ("ollama", "ollama/qwen3", "none"),
    ("ollama", "ollama/qwen3", "low"),
    ("ollama", "ollama_chat/qwen3", "none"),
])
def test_wire_body(endpoint, path, backend, model, effort):
    base, requests = endpoint
    base += "/v1" if model.startswith("openai/") else ""
    kwargs = reasoning_kwargs(model, backend, effort)
    if path == "direct":
        response = LiteLLMClient(model, api_base=base, api_key="test", timeout_seconds=5,
                                reasoning_backend=backend, reasoning_effort=effort).complete(
            [{"role": "user", "content": "hello"}])
        assert response.content == "ok"
    else:
        response = ChatLiteLLM(model=model, api_base=base, api_key="test",
                              request_timeout=5, model_kwargs=kwargs).invoke("hello")
        assert response.content == "ok"
    body = [body for url, body in requests if url != "/api/show"][-1]
    if effort is None:
        assert "reasoning_effort" not in body and "think" not in body
    elif model.startswith(("ollama/", "ollama_chat/")):
        assert body["think"] == (False if effort == "none" else effort)
    else:
        assert body["reasoning_effort"] == effort


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high"])
def test_config_accepts_efforts(tmp_path, effort):
    profile = _parse_agent_profile({"id": "a", "redmine_user_id": 1, "redmine_api_key": "k",
        "llm_model": "openai/qwen3", "context_window_tokens": 1000, "llm_api_key": "",
        "llm_reasoning_backend": "strata", "llm_reasoning_effort": effort},
        index=1, config_path=tmp_path / "agents.toml")
    assert profile.llm_reasoning_effort == effort


@pytest.mark.parametrize("model,backend,effort", [
    ("openai/a", None, "low"), ("openai/a", "strata", None),
    ("openai/a", "bad", "low"), ("openai/a", "strata", "bad"),
    ("lm_studio/a", "strata", "low"), ("ollama/a", "strata", "none"),
])
def test_invalid_combinations(model, backend, effort):
    with pytest.raises(ConfigError):
        reasoning_kwargs(model, backend, effort)


@pytest.mark.parametrize("values,effort,valid", [
    ([False, True], "none", True), ([False, True], "low", False),
    (["low", "medium", "high"], "low", True), (["low"], "none", False),
    (None, "low", False),
])
def test_ollama_capabilities(monkeypatch, values, effort, valid):
    def post(url, **kwargs):
        assert url == "http://localhost:11434/api/show"
        assert kwargs["json"] == {"model": "qwen3"}
        return httpx.Response(200, json={"thinking": {"values": values}},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr("taskboard_agent.reasoning.httpx.post", post)
    if valid:
        validate_ollama_capabilities("openai/qwen3", "http://localhost:11434/v1", effort)
    else:
        with pytest.raises(ConfigError):
            validate_ollama_capabilities("openai/qwen3", "http://localhost:11434/v1", effort)


def test_capability_fetch_failure(monkeypatch):
    def post(*args, **kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr("taskboard_agent.reasoning.httpx.post", post)
    with pytest.raises(ConfigError, match="Unable to verify"):
        validate_ollama_capabilities("ollama/qwen3", "http://localhost:11434", "none")


def test_non_supported_request_does_not_drop_effort(monkeypatch):
    calls = []
    def completion(**kwargs):
        calls.append(kwargs)
        raise ValueError("unsupported reasoning_effort")
    monkeypatch.setattr("taskboard_agent.llm.litellm.completion", completion)
    with pytest.raises(LLMError):
        LiteLLMClient("openai/a", reasoning_backend="openai_compatible",
                      reasoning_effort="low").complete([{"role": "user", "content": "hi"}])
    assert len(calls) == 1 and calls[0]["reasoning_effort"] == "low"


@pytest.mark.parametrize("value", [True, 1, "", " ", "xhigh"])
def test_config_rejects_invalid_effort(value):
    with pytest.raises(ConfigError):
        _parse_agent_profile({"id": "a", "redmine_user_id": 1, "redmine_api_key": "k",
            "llm_model": "openai/qwen3", "context_window_tokens": 1000, "llm_api_key": "",
            "llm_reasoning_backend": "strata", "llm_reasoning_effort": value},
            index=1, config_path=Path("agents.toml"))


def test_runtime_rejects_unsupported_ollama_before_side_effects(monkeypatch):
    from taskboard_agent.cli import build_runtime
    profile = SimpleNamespace(llm_model="ollama/qwen3", llm_reasoning_backend="ollama",
        llm_reasoning_effort="low", llm_api_base="http://localhost:11434",
        llm_timeout_seconds=None, llm_api_key="")
    monkeypatch.setattr("taskboard_agent.cli.load_config", lambda **kw: SimpleNamespace(agents=[profile]))
    def post(url, **kwargs):
        return httpx.Response(200, json={"thinking": {"values": [True, False]}},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr("taskboard_agent.reasoning.httpx.post", post)
    with pytest.raises(ConfigError, match="does not support"):
        with build_runtime(dry_run=False):
            pytest.fail("runtime must not be constructed")
