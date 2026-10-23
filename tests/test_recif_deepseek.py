"""DeepSeek judge tests: no real credentials, network calls, or GPU required."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback
from types import ModuleType, SimpleNamespace

import pytest

from real_quant.full_precision import recif_wip as wip
from real_quant.full_precision import run_recif_baseline as runner
from real_quant.full_precision.recif_tasks import atomic_json, digest


EXAMPLE = Path(__file__).resolve().parents[1] / "real_quant/full_precision/recif_judge.deepseek.example.json"
SECRET = "unit-test-secret-not-a-real-key"
POINT = {"info_point": "a cat", "importance_score": 5}
MATCH = {"matches": [{"gt_wip": POINT, "model_wip": POINT}],
         "unmatched_gt_wips": [], "unmatched_model_wips": []}


def response(text='{"wips": []}', finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
                           message=SimpleNamespace(content=text, reasoning_content="unused reasoning"))])


class HTTPError(RuntimeError):
    def __init__(self, status):
        super().__init__("service echoed credential: " + SECRET)
        self.status_code = status


@pytest.fixture
def sdk(monkeypatch):
    state = SimpleNamespace(responses=[], requests=[], inits=[], delays=[])
    module = ModuleType("openai")
    module.APIConnectionError = type("APIConnectionError", (RuntimeError,), {})
    module.APITimeoutError = type("APITimeoutError", (module.APIConnectionError,), {})

    def create(**kwargs):
        state.requests.append(kwargs)
        value = state.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def client(**kwargs):
        state.inits.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    module.OpenAI = client
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    monkeypatch.setattr(wip.time, "sleep", state.delays.append)
    state.module = module
    return state


def config_args(tmp_path, config=None):
    path = tmp_path / "judge.json"
    atomic_json(path, json.loads(EXAMPLE.read_text()) if config is None else config)
    return SimpleNamespace(allow_judge=True, judge_config=path)


def preflight(tmp_path, monkeypatch, config=None):
    monkeypatch.setattr(wip.importlib.util, "find_spec", lambda name: object())
    return wip.judge_preflight(config_args(tmp_path, config))


def test_deepseek_preflight_only_loads_its_own_dependencies(tmp_path, monkeypatch, sdk):
    names = []
    monkeypatch.setattr(wip.importlib.util, "find_spec", lambda name: names.append(name) or object())
    config = wip.judge_preflight(config_args(tmp_path))
    assert names == ["bert_score", "openai"]
    assert config["api_key_env"] == "DEEPSEEK_API_KEY"
    assert SECRET not in json.dumps(config)
    assert not sdk.inits and not sdk.requests


def test_old_gemini_config_is_unchanged(tmp_path, monkeypatch):
    legacy = {"project": "test-project", "location": "us-central1", "model_name": "gemini-test",
              "temperature": .01, "max_new_tokens": 10000, "max_retries": 3, "retry_delay": 2}
    assert preflight(tmp_path, monkeypatch, legacy) == legacy
    assert preflight(tmp_path, monkeypatch, {"gemini": legacy}) == legacy


def test_missing_key_fails_without_showing_any_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        preflight(tmp_path, monkeypatch)


@pytest.mark.parametrize("field,value", [
    ("provider", "unknown"),
    ("base_url", "https://user:" + SECRET + "@example.com"),
    ("base_url", "https://example.com?api_key=" + SECRET),
    ("base_url", "http://api.deepseek.com"),
    ("api_key_env", SECRET), ("thinking", True), ("timeout_seconds", 0),
    ("max_retries", 0), ("max_new_tokens", 1.5), ("temperature", float("nan")),
])
def test_invalid_or_secret_config_is_rejected(tmp_path, monkeypatch, sdk, field, value):
    config = json.loads(EXAMPLE.read_text())
    config[field] = value
    # JSON's NaN extension is deliberately tested; preflight must reject it.
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError) as error:
        wip.judge_preflight(SimpleNamespace(allow_judge=True, judge_config=path))
    assert SECRET not in str(error.value)
    assert not sdk.requests


@pytest.mark.parametrize("env_present", [False, True])
def test_file_key_works_without_env_and_takes_precedence(tmp_path, monkeypatch, sdk, env_present):
    config = json.loads(EXAMPLE.read_text())
    config.pop("api_key_env")
    config["api_key"] = SECRET
    if env_present:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "different-unit-test-key")
    else:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = preflight(tmp_path, monkeypatch, config)
    client = wip.create_judge(config)
    assert sdk.inits[0]["api_key"] == SECRET
    assert client.config == wip.public_judge_config(config)
    assert "api_key" not in client.config
    assert SECRET not in json.dumps(client.config)
    assert not sdk.requests


@pytest.mark.parametrize("value", [None, 123, {}, "", "YOUR_DEEPSEEK_API_KEY", SECRET + " "])
def test_invalid_file_key_does_not_fall_back_to_env(tmp_path, monkeypatch, sdk, value):
    config = json.loads(EXAMPLE.read_text())
    config["api_key"] = value
    with pytest.raises(ValueError, match="private judge_config") as error:
        preflight(tmp_path, monkeypatch, config)
    assert SECRET not in str(error.value)
    assert not sdk.inits and not sdk.requests


def test_missing_deepseek_dependency_has_correct_install_hint(tmp_path, monkeypatch, sdk):
    monkeypatch.setattr(wip.importlib.util, "find_spec", lambda name: None if name == "openai" else object())
    with pytest.raises(ImportError, match="requirements-recif-deepseek.txt"):
        wip.judge_preflight(config_args(tmp_path))


@pytest.mark.parametrize("thinking", ["disabled", "enabled"])
def test_request_parameters_and_final_content(tmp_path, monkeypatch, sdk, thinking):
    config = preflight(tmp_path, monkeypatch)
    config["thinking"] = thinking
    sdk.responses = [response()]
    client = wip.create_judge(config)
    assert client.generate("Return JSON, using the original WIP prompt.") == '{"wips": []}'
    request = sdk.requests[0]
    assert request["model"] == config["model_name"]
    assert request["extra_body"] == {"thinking": {"type": thinking}}
    assert request["response_format"] == {"type": "json_object"}
    assert request["max_tokens"] == 10000
    assert request["stream"] is False
    assert ("temperature" in request) == (thinking == "disabled")
    assert sdk.inits[0]["api_key"] == SECRET
    assert sdk.inits[0]["max_retries"] == 0
    assert SECRET not in json.dumps(request)


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_nonretryable_errors_stop_and_never_echo_credentials(tmp_path, monkeypatch, sdk, capsys, status):
    client = wip.create_judge(preflight(tmp_path, monkeypatch))
    sdk.responses = [HTTPError(status), response()]
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
        client.generate("Return JSON.")
    assert len(sdk.requests) == 1
    assert not sdk.delays
    assert SECRET not in "".join(traceback.format_exception(type(error.value), error.value, error.tb))
    assert SECRET not in capsys.readouterr().out


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
def test_retry_transient_http_failures(tmp_path, monkeypatch, sdk, status):
    client = wip.create_judge(preflight(tmp_path, monkeypatch))
    sdk.responses = [HTTPError(status), response()]
    assert client.generate("Return JSON.") == '{"wips": []}'
    assert len(sdk.requests) == 2
    assert sdk.delays == [2]


def test_connection_failure_retry_limit(tmp_path, monkeypatch, sdk, capsys):
    client = wip.create_judge(preflight(tmp_path, monkeypatch))
    sdk.responses = [sdk.module.APITimeoutError(SECRET) for _ in range(3)]
    with pytest.raises(RuntimeError, match=r"after 3 attempt"):
        client.generate("Return JSON.")
    assert len(sdk.requests) == 3
    assert sdk.delays == [2, 4]
    assert SECRET not in capsys.readouterr().out


@pytest.mark.parametrize("text", ["", None, "bad JSON", "[]"])
def test_empty_and_invalid_json_are_retried(tmp_path, monkeypatch, sdk, text):
    client = wip.create_judge(preflight(tmp_path, monkeypatch))
    sdk.responses = [response(text), response()]
    assert client.generate("Return JSON.") == '{"wips": []}'
    assert len(sdk.requests) == 2


def test_truncation_stops_instead_of_repeating_same_token_limit(tmp_path, monkeypatch, sdk):
    client = wip.create_judge(preflight(tmp_path, monkeypatch))
    sdk.responses = [response('{"wips": [', finish="length")]
    with pytest.raises(RuntimeError, match="increase max_new_tokens"):
        client.generate("Return JSON.")
    assert len(sdk.requests) == 1


def test_missing_json_instruction_causes_no_request(tmp_path, monkeypatch, sdk):
    client = wip.create_judge(preflight(tmp_path, monkeypatch))
    with pytest.raises(ValueError, match="JSON instructions"):
        client.generate("Unstructured request")
    assert not sdk.requests


@pytest.mark.parametrize("auth_source", ["env", "file"])
def test_score_pipeline_cache_and_secret_free_results(tmp_path, monkeypatch, sdk, auth_source):
    from benchmark.tasks.v1_0.item_understand import utils

    def fake_bert(ids, matches, *args):
        for result in matches.values():
            for match in result["matches"]:
                match["match_quality"] = 1.0

    monkeypatch.setattr(utils, "_compute_bertscore_incremental", fake_bert)
    monkeypatch.setattr(wip.importlib.util, "find_spec", lambda name: object())
    config = json.loads(EXAMPLE.read_text())
    if auth_source == "file":
        config["api_key"] = SECRET
        monkeypatch.delenv("DEEPSEEK_API_KEY")
    args = config_args(tmp_path, config)
    args.task = "item_understand"
    args.data_dir = tmp_path
    args.bertscore_model = "test-bert"
    args.wip_reference_cache = tmp_path / "references"
    directory = tmp_path / "results"
    protocol = {"sample_count": 1}
    samples = {"0": {"ground_truth": "cat", "generations": ["a cat"]}}
    atomic_json(directory / "run_config.json", protocol)
    atomic_json(directory / "test_generated.json", {"run_id": digest(protocol), "model_name": "1.7B",
                "task_name": "item_understand", "status": "generation_complete", "samples": samples})
    sdk.responses = [response(json.dumps({"wips": [POINT]})),
                     response(json.dumps({"wips": [POINT]})), response(json.dumps(MATCH))]
    runner.score(args, directory)
    result = json.loads((directory / "eval_results.json").read_text())
    assert result["metrics"]["macro_wip_double_weighted_f1"] == 1
    assert result["score_config"]["judge"]["provider"] == "deepseek"
    assert len(sdk.requests) == 3
    for path in tmp_path.rglob("*.json"):
        if path != args.judge_config:
            assert SECRET not in path.read_text()
    if auth_source == "file":
        config["api_key"] = "rotated-unit-test-key"
        atomic_json(args.judge_config, config)
    else:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "rotated-unit-test-key")
    runner.score(args, directory)
    assert len(sdk.requests) == 3  # Changing the secret does not invalidate scores.
    # Also exercise WIP caches directly (not just the final eval_results shortcut).
    private_config = wip.judge_preflight(args)
    wip.score_wip(samples, directory, private_config, args.bertscore_model,
                  reference_cache_dir=args.wip_reference_cache)
    assert len(sdk.requests) == 3
    assert len(list((directory / "wip_cache").iterdir())) == 1
    # New prediction cache must reuse the shared reference extraction after rotation.
    sdk.responses = [response(json.dumps({"wips": [POINT]})), response(json.dumps(MATCH))]
    wip.score_wip(samples, tmp_path / "other_results", private_config, args.bertscore_model,
                  reference_cache_dir=args.wip_reference_cache)
    assert len(sdk.requests) == 5
    assert len(list(args.wip_reference_cache.glob("*.json"))) == 1


def test_gemini_factory_still_works(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(wip, "GeminiJudge", lambda config: sentinel)
    assert wip.create_judge({}) is sentinel
    assert wip.create_judge({"provider": "gemini"}) is sentinel


@pytest.mark.parametrize("first_status", [200, 401, 429])
@pytest.mark.parametrize("auth_source", ["env", "file"])
def test_real_sdk_offline_transport(monkeypatch, first_status, auth_source):
    """Optional integration test; httpx intercepts every request (never network)."""
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    original_client = openai.OpenAI
    config = json.loads(EXAMPLE.read_text())
    requests = []
    delays = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    if auth_source == "file":
        config["api_key"] = SECRET
        monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.setattr(wip.time, "sleep", delays.append)

    def handler(request):
        requests.append(request)
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert request.headers["authorization"] == "Bearer " + SECRET
        body = json.loads(request.content)
        assert body["thinking"] == {"type": "disabled"}
        assert body["response_format"] == {"type": "json_object"}
        assert body["temperature"] == .01
        assert SECRET not in request.content.decode()
        if len(requests) == 1 and first_status != 200:
            return httpx.Response(first_status, json={"error": {"message": SECRET, "type": "test_error"}})
        return httpx.Response(200, json={"id": "offline-test", "object": "chat.completion", "created": 0,
                              "model": config["model_name"], "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": '{"wips": []}'}}]})

    def client(**kwargs):
        return original_client(http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)

    monkeypatch.setattr(openai, "OpenAI", client)
    judge = wip.create_judge(config)
    try:
        if first_status == 401:
            with pytest.raises(RuntimeError, match="HTTP 401") as error:
                judge.generate("Return JSON.")
            assert SECRET not in "".join(traceback.format_exception(type(error.value), error.value, error.tb))
            assert len(requests) == 1
        else:
            assert judge.generate("Return JSON.") == '{"wips": []}'
            assert len(requests) == (2 if first_status == 429 else 1)
        assert delays == ([2] if first_status == 429 else [])
    finally:
        judge.client.close()
