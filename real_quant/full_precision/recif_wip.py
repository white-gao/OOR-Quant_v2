"""Strict, resumable orchestration of OpenOneRec's WIP prompts and metrics.

We reuse the retained official functions, not a replacement caption metric.
API/parse/BERTScore failures raise instead of silently dropping scored rows.
Valid empty WIP extractions retain the upstream exclusion convention, with
explicit coverage counters. External judge use is opt-in at the CLI.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from tqdm.auto import tqdm

from .recif_tasks import atomic_json, digest


def judge_preflight(args):
    if not args.allow_judge or not args.judge_config:
        raise ValueError("WIP requires --allow_judge --judge_config PATH (external, potentially paid API).")
    config = json.loads(args.judge_config.read_text())
    if not isinstance(config, dict):
        raise ValueError("judge_config must be a JSON object.")
    # Preserve the old flat Gemini config and the official Gemini section.
    if "provider" not in config and "gemini" in config:
        config = config["gemini"]
    if not isinstance(config, dict):
        raise ValueError("Judge provider configuration must be a JSON object.")
    provider = config.get("provider", "gemini")
    if provider not in ("gemini", "deepseek"):
        raise ValueError("Judge provider must be gemini or deepseek.")
    common = {"provider", "model_name", "temperature", "max_new_tokens", "max_retries", "retry_delay"}
    allowed = common | ({"project", "location", "credentials_path"} if provider == "gemini" else
                        {"base_url", "api_key", "api_key_env", "thinking", "timeout_seconds"})
    if set(config) - allowed:
        # Never echo raw config values: a misplaced API key may be among them.
        raise ValueError("Unsupported judge config fields. Use the provider's example JSON.")
    required = ("project", "location", "model_name") if provider == "gemini" else ("model_name",)
    for key in required:
        if not isinstance(config.get(key), str) or not config[key].strip() or config[key].startswith("YOUR_"):
            raise ValueError(f"Set {provider} {key} in judge_config; see the provider's example JSON.")
    config.setdefault("temperature", 0.01)
    config.setdefault("max_new_tokens", 10000)
    config.setdefault("max_retries", 3)
    config.setdefault("retry_delay", 2)
    for key in ("max_retries", "max_new_tokens"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("temperature", "retry_delay"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if config["temperature"] > 2:
        raise ValueError("temperature must be at most 2.")
    if provider == "gemini":
        if config.get("credentials_path") and not Path(config["credentials_path"]).expanduser().is_file():
            raise FileNotFoundError("Gemini credentials_path does not exist.")
        modules = ("bert_score", "google.genai")
        requirements = "requirements-recif-judge.txt"
    else:
        config.setdefault("base_url", "https://api.deepseek.com")
        config.setdefault("api_key_env", "DEEPSEEK_API_KEY")
        config.setdefault("thinking", "disabled")
        config.setdefault("timeout_seconds", 120)
        validate_deepseek_config(config)
        deepseek_api_key(config)  # Validate before any model load or API request.
        modules = ("bert_score", "openai")
        requirements = "requirements-recif-deepseek.txt"
    for module in modules:
        try:
            available = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            available = False
        if not available:
            raise ImportError(f"Missing {module}; install real_quant/full_precision/{requirements}")
    return config


def validate_deepseek_config(config):
    if config.get("provider") != "deepseek":
        raise ValueError("DeepSeek configuration requires provider=deepseek.")
    # Disallow embedded URL credentials/query tokens from being persisted in
    # score_config. Credentials must use api_key or the named environment variable.
    if not isinstance(config["base_url"], str):
        raise ValueError("DeepSeek base_url must be an HTTPS endpoint string.")
    try:
        url = urlsplit(config["base_url"])
        valid_url = (url.scheme == "https" and bool(url.hostname) and not url.username
                     and not url.password and not url.query and not url.fragment)
        _ = url.port
    except (TypeError, ValueError):
        valid_url = False
    if not valid_url or any(c.isspace() for c in config["base_url"]):
        raise ValueError("DeepSeek base_url must be an HTTPS endpoint without credentials, query, or fragment.")
    if not isinstance(config["api_key_env"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config["api_key_env"]):
        raise ValueError("api_key_env must name an environment variable, not contain an API key.")
    if config["thinking"] not in ("enabled", "disabled"):
        raise ValueError("thinking must be enabled or disabled.")
    if (type(config["timeout_seconds"]) not in (int, float)
            or not math.isfinite(config["timeout_seconds"]) or config["timeout_seconds"] <= 0):
        raise ValueError("timeout_seconds must be positive and finite.")


def deepseek_api_key(config):
    # An explicit file value takes precedence; a typo/placeholder must not
    # silently fall back to another account's environment credential.
    inline = "api_key" in config
    key = config["api_key"] if inline else os.environ.get(config.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    if not isinstance(key, str) or not key or key.startswith("YOUR_") or any(c.isspace() for c in key):
        if inline:
            raise ValueError("Set a valid api_key in your private judge_config file; replace the placeholder.")
        raise ValueError("Set a valid API key via api_key in a private judge_config file or via "
                         "api_key_env (default: DEEPSEEK_API_KEY).")
    return key


def public_judge_config(config):
    """Persist/cache only non-secret settings; key rotation is not a new judge."""
    return {key: value for key, value in config.items() if key != "api_key"}


class _JudgeResponseError(ValueError):
    def __init__(self, message, *, retryable=True):
        super().__init__(message)
        self.retryable = retryable


class DeepSeekJudge:
    """DeepSeek Chat Completions; no key, reasoning trace, or raw error is logged."""

    def __init__(self, config):
        import openai
        self.config = public_judge_config(config)
        self._connection_errors = (openai.APIConnectionError, openai.APITimeoutError)
        # Disable SDK retries so max_retries is the actual total attempt limit.
        self.client = openai.OpenAI(api_key=deepseek_api_key(config), base_url=config["base_url"],
                                    timeout=config["timeout_seconds"], max_retries=0)

    @staticmethod
    def _content(response):
        if not response.choices:
            raise _JudgeResponseError("no choices returned")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise _JudgeResponseError("JSON output was truncated; increase max_new_tokens in judge_config",
                                      retryable=False)
        if choice.finish_reason != "stop":
            raise _JudgeResponseError("response did not finish normally", retryable=False)
        text = choice.message.content  # Do not concatenate reasoning_content.
        if not isinstance(text, str) or not text.strip():
            raise _JudgeResponseError("empty final content")
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            raise _JudgeResponseError("invalid JSON output") from None
        if not isinstance(parsed, dict):
            raise _JudgeResponseError("expected a JSON object")
        return text

    def generate(self, prompt):
        kwargs = {
            "model": self.config["model_name"], "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config["max_new_tokens"], "stream": False,
            "response_format": {"type": "json_object"},
            "extra_body": {"thinking": {"type": self.config["thinking"]}},
        }
        if self.config["thinking"] == "disabled":
            kwargs["temperature"] = self.config["temperature"]
        # Official WIP prompts already contain JSON instructions and examples.
        if "json" not in prompt.lower():
            raise ValueError("DeepSeek JSON mode requires explicit JSON instructions in the prompt.")
        for attempt in range(self.config["max_retries"]):
            try:
                return self._content(self.client.chat.completions.create(**kwargs))
            except _JudgeResponseError as exc:
                message, retryable = str(exc), exc.retryable
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                connection_error = isinstance(exc, self._connection_errors)
                retryable = connection_error or (isinstance(status, int) and
                                                 (status in (408, 409, 429) or 500 <= status <= 599))
                message = (f"HTTP {status}" if isinstance(status, int) else
                           "connection/timeout failure" if connection_error else "client request failed")
                if not retryable:
                    message += "; check API key, balance, endpoint, and model configuration"
            if not retryable or attempt + 1 == self.config["max_retries"]:
                # Do not expose exception bodies/headers: services may echo secrets.
                raise RuntimeError(f"DeepSeek judge failed after {attempt + 1} attempt(s): {message}") from None
            print(f"[judge retry] DeepSeek {message}; attempt {attempt + 2}/{self.config['max_retries']}", flush=True)
            time.sleep(min(30, self.config["retry_delay"] * 2 ** attempt))


def create_judge(config):
    provider = config.get("provider", "gemini")
    if provider == "deepseek":
        return DeepSeekJudge(config)
    if provider == "gemini":
        return GeminiJudge(config)
    raise ValueError("Judge provider must be gemini or deepseek.")


class GeminiJudge:
    """Official judge protocol via Google Gen AI (legacy Vertex SDK retired)."""

    def __init__(self, config):
        from google import genai
        from google.genai import types
        if config.get("credentials_path"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path(config["credentials_path"]).expanduser())
        self.client = genai.Client(vertexai=True, project=config["project"], location=config["location"],
                                   http_options=types.HttpOptions(api_version="v1", timeout=120000))
        self.config = config

    def generate(self, prompt):
        for attempt in range(self.config["max_retries"]):
            try:
                response = self.client.models.generate_content(
                    model=self.config["model_name"], contents=prompt,
                    config={"temperature": self.config["temperature"],
                            "max_output_tokens": self.config["max_new_tokens"]})
                if not response.text:
                    raise RuntimeError("Empty judge response.")
                return response.text
            except Exception:
                if attempt + 1 == self.config["max_retries"]:
                    raise
                time.sleep(min(30, self.config["retry_delay"] * 2 ** attempt))


def validate_wips(value):
    if not isinstance(value, list):
        raise ValueError("Judge WIPs must be a list.")
    for wip in value:
        if (not isinstance(wip, dict) or not isinstance(wip.get("info_point"), str)
                or not wip["info_point"].strip()
                or not isinstance(wip.get("importance_score"), (int, float))
                or not math.isfinite(wip["importance_score"]) or not 1 <= wip["importance_score"] <= 5):
            raise ValueError("Invalid WIP text/importance_score in judge response.")


def validate_match(result):
    if not isinstance(result, dict) or not isinstance(result.get("matches"), list):
        raise ValueError("Invalid WIP matching response.")
    validate_wips(result.get("unmatched_gt_wips"))
    validate_wips(result.get("unmatched_model_wips"))
    for match in result["matches"]:
        validate_wips([match.get("gt_wip"), match.get("model_wip")])


def score_wip(samples, directory, judge_config, bertscore_model, *, client=None, bertscore_fn=None,
              reference_cache_dir=None):
    from benchmark.tasks.v1_0.item_understand import utils
    if not samples:
        raise ValueError("Cannot evaluate an empty WIP dataset.")
    # Hash texts/config: a different model output or judge cannot reuse stale WIPs.
    public_config = public_judge_config(judge_config)
    cache_id = digest({"samples": samples, "judge": public_config,
                       "bertscore": [bertscore_model, 9], "protocol": 1})
    cache_dir = directory / "wip_cache" / cache_id
    reference_cache_dir = reference_cache_dir or directory / "reference_cache"
    matches = {}
    per_sample = {}
    states = {}
    for key, sample in tqdm(samples.items(), desc="WIP extract / match", unit="sample"):
        cache_file = cache_dir / f"{int(key):09d}.json"
        state = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        texts = {"reference_wips": sample["ground_truth"],
                 "prediction_wips": utils.extract_after_think(sample["generations"][0])}
        for field, text in texts.items():
            if field not in state:
                shared = reference_cache_dir / (digest({"text": text, "judge": public_config,
                                                        "protocol": 1}) + ".json")
                if field == "reference_wips" and shared.exists():
                    value = json.loads(shared.read_text())
                else:
                    if client is None:
                        client = create_judge(judge_config)
                    value, error = utils.extract_wips_single(text, client)
                    if error or value is None:
                        raise RuntimeError(f"WIP extraction failed at {key}/{field}: {error}")
                validate_wips(value)
                if field == "reference_wips" and not shared.exists():
                    atomic_json(shared, value)
                state[field] = value
                atomic_json(cache_file, state)
            validate_wips(state[field])
            if field == "reference_wips" and not state[field]:
                break  # Upstream does not extract predictions without GT WIPs.
        if not state["reference_wips"] or not state.get("prediction_wips"):
            per_sample[key] = {"status": "excluded_empty_wips"}
            states[key] = state
            continue
        if "match_result" not in state:
            if client is None:
                client = create_judge(judge_config)
            result, error = utils.match_wips_single(state["reference_wips"], state["prediction_wips"], client)
            if error or result is None:
                raise RuntimeError(f"WIP matching failed at {key}: {error}")
            validate_match(result)
            state["match_result"] = result
            atomic_json(cache_file, state)
        validate_match(state["match_result"])
        matches[key] = state["match_result"]
        states[key] = state
    if not matches:
        raise RuntimeError("No scoreable WIP samples; no valid metric can be reported.")
    compute_bert = bertscore_fn or utils._compute_bertscore_incremental
    compute_bert(list(matches), matches, bertscore_model, 9)
    for key, result in matches.items():
        if any(not isinstance(m.get("match_quality"), (int, float))
               or not math.isfinite(m["match_quality"]) for m in result["matches"]):
            raise RuntimeError(f"Missing/non-finite BERTScore at sample {key}; no partial metric saved.")
        states[key]["match_result"] = result
        atomic_json(cache_dir / f"{int(key):09d}.json", states[key])
    metrics = {"num_samples": len(samples), "wip_num_samples": len(matches),
               "wip_excluded_empty_samples": len(samples) - len(matches),
               "wip_coverage": len(matches) / len(samples)}
    for family, fn in (("unweighted", utils.calculate_unweighted_metrics),
                       ("importance_weighted", utils.calculate_importance_weighted_metrics),
                       ("double_weighted", utils.calculate_double_weighted_metrics)):
        result = fn(matches, core_threshold=5)
        metrics.update({k: v for k, v in result.items() if k != "per_sample"})
        for key, values in result["per_sample"].items():
            per_sample.setdefault(key, {"status": "scored"}).update({
                f"wip_{family}_f1": values["overall_f1"], f"wip_{family}_core_f1": values["core_f1"]})
    if not all(math.isfinite(value) for key, value in metrics.items() if key.startswith("macro_")):
        raise RuntimeError("Non-finite WIP metrics.")
    if len(matches) != len(samples):
        print(f"[coverage] Official WIP mean uses {len(matches)}/{len(samples)} samples; valid empty WIPs are excluded.", flush=True)
    return metrics, per_sample
