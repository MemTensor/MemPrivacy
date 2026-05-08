import re
import logging
import traceback
from typing import Dict, Any, Optional

import yaml
import json_repair
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)
from openai import OpenAI


logger = logging.getLogger(__name__)


def _load_config(config_path: str = "eval_config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Module-level cache: config and OpenAI client instances
# ---------------------------------------------------------------------------
_CONFIG: Optional[Dict[str, Any]] = None
_CLIENTS: Dict[str, OpenAI] = {}          # key = (base_url, api_key)


def _get_config(config_path: str = "eval_config.yaml") -> Dict[str, Any]:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config(config_path)
    return _CONFIG


def _get_client(base_url: str, api_key: str) -> OpenAI:
    """Cache client by (base_url, api_key) to avoid repeated creation."""
    key = (base_url, api_key)
    if key not in _CLIENTS:
        _CLIENTS[key] = OpenAI(base_url=base_url, api_key=api_key)
    return _CLIENTS[key]


def _resolve_llm_params(
    llm_type: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve the full set of request parameters and client info from config
    based on the given llm_type.
    If a specific llm section has its own non-empty base_url / api_key,
    those values are used; otherwise fall back to the top-level
    openai_base_url / openai_api_key.
    """
    if llm_type not in config:
        raise ValueError(
            f"Unknown llm_type '{llm_type}'. "
            f"Available types: {[k for k in config if k.endswith('_llm')]}"
        )

    llm_cfg: Dict[str, Any] = config[llm_type]

    # base_url / api_key: prefer the llm section's own config
    base_url = llm_cfg.get("base_url") or config.get("openai_base_url", "")
    api_key = llm_cfg.get("api_key") or config.get("openai_api_key", "")

    if not base_url or not api_key:
        raise ValueError(
            f"base_url or api_key is missing for llm_type '{llm_type}'. "
            "Please set them in the llm section or at the top level of the config."
        )

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": llm_cfg["model"],
        "max_tokens": llm_cfg.get("max_tokens"),
        "temperature": llm_cfg.get("temperature"),
        "timeout": llm_cfg.get("timeout"),
        "retry_times": llm_cfg.get("retry_times", 3),
        "wait_time_lower": llm_cfg.get("wait_time_lower", 10),
        "wait_time_upper": llm_cfg.get("wait_time_upper", 30),
    }


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def call_llm(
    query: str,
    llm_type: str = "answer_llm",
    config_path: str = "eval_config.yaml",
    return_parsed_json: bool = False,
    extract_json: bool = False,
) -> str:
    """
    Send a request to the LLM and return the result.

    Parameters
    ----------
    query : str
        The full content to send to the LLM (used as the user message).
    llm_type : str
        The LLM type key in the config, e.g. "memory_llm", "answer_llm",
        "judgment_llm", "privacy_llm", etc. New types can be added to
        eval_config.yaml and will be supported automatically.
    config_path : str
        Path to the config file. Defaults to "eval_config.yaml".
    return_parsed_json : bool
        Whether to attempt parsing the response content as JSON (dict / list).
    extract_json : bool
        If False, return raw text without JSON parsing;
        if True, behaviour depends on return_parsed_json.

    Returns
    -------
    str | dict | list
        The LLM response content. The exact type depends on extract_json
        and return_parsed_json parameters.
    """
    config = _get_config(config_path)
    params = _resolve_llm_params(llm_type, config)

    # Build a retry-wrapped inner request function using per-llm retry params
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(
            min=params["wait_time_lower"],
            max=params["wait_time_upper"],
        ),
        stop=stop_after_attempt(params["retry_times"]),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _request():
        client = _get_client(params["base_url"], params["api_key"])

        messages = [{"role": "user", "content": query}]

        request_kwargs: Dict[str, Any] = {
            "model": params["model"],
            "messages": messages,
        }
        if params.get("max_tokens") is not None:
            request_kwargs["max_tokens"] = params["max_tokens"]
        if params.get("temperature") is not None:
            request_kwargs["temperature"] = params["temperature"]
        if params.get("timeout") is not None:
            request_kwargs["timeout"] = params["timeout"]

        response = client.chat.completions.create(**request_kwargs)
        content: str = response.choices[0].message.content.strip()
        logger.debug(f"[{llm_type}] API response content: {content}")

        # Strip possible <think>...</think> blocks
        content = re.sub(
            r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE
        ).strip()

        return content

    content = _request()

    # ---- Post-processing ----
    if not extract_json:
        return content

    if return_parsed_json:
        try:
            repaired = json_repair.loads(content)
            assert isinstance(repaired, (dict, list)), \
                "Parsed JSON must be a dict or list"
        except Exception as e:
            logger.error(
                f"Failed to parse JSON: {e}, {traceback.format_exc()}"
            )
            raise ValueError(
                f"Failed to parse JSON from content: {content[:200]}..."
            )
        return repaired

    return content


def verify_mcq_answer(
    response: str,
    answer: str
) -> bool:
    pattern = re.compile(r'^[\(<{]?([abcd])[\)>}]?$', re.IGNORECASE)

    r_match = pattern.match(response.strip())
    a_match = pattern.match(answer.strip())

    if not r_match or not a_match:
        return False, False

    return r_match.group(1).lower() == a_match.group(1).lower(), True


if __name__ == "__main__":
    # 1. Simplest call — plain text response
    result = call_llm(
        query="who are you?",
        llm_type="answer_llm",
    )
    print(result)

    # 2. Request JSON response
    result = call_llm(
        query='Return your answer in JSON format: {"score": ..., "reason": ...}',
        llm_type="judgment_llm",
        extract_json=True,
        return_parsed_json=True,
    )
    print(result)  # dict