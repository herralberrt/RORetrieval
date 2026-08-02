"""
The single place where an LLM is called.

QUICK START
-----------
1. Copy the example config:      cp .env.example .env
2. Put ONE api key in .env:

       ANTHROPIC_API_KEY=sk-ant-...      # for Claude
   or
       OPENAI_API_KEY=sk-...             # for OpenAI

3. That's it. Everything that needs an LLM goes through this module:

       from llm_client import get_client
       client = get_client()
       if client.available:
           text = client.complete("Salut!")

If no key is set, `client.available` is False and every caller falls back to
its template/offline path instead of crashing.

Supported providers: "anthropic" (Claude) and "openai". The provider is picked
automatically from whichever key is present; set LLM_PROVIDER in .env to force
one when both keys exist.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_config


# Default models per provider. Override with ANTHROPIC_MODEL / OPENAI_MODEL.
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o-mini",
}

# Claude models that reject `temperature` / `top_p` / `top_k` with a 400.
# On these, thinking depth is controlled with output_config.effort instead.
_NO_SAMPLING_PARAMS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)

# Placeholder values in .env.example — treated as "no key set".
_PLACEHOLDER_PREFIXES = ("sk-your", "sk-ant-your", "org-your", "your-", "hf_your")


def _looks_like_placeholder(key: Optional[str]) -> bool:
    if not key:
        return True
    return key.strip().lower().startswith(_PLACEHOLDER_PREFIXES)


def load_env(env_file: Optional[str] = None) -> Dict[str, str]:
    """
    Read config from .env (searching up from this file to the project root),
    falling back to real environment variables.
    """
    if env_file is None:
        here = Path(__file__).resolve()
        for parent in [here.parent] + list(here.parents):
            candidate = parent / ".env"
            if candidate.exists():
                env_file = str(candidate)
                break

    config = get_config(env_file) if env_file else {}

    # Real environment variables win over .env.
    for key in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_ORG_ID",
        "ANTHROPIC_MODEL", "OPENAI_MODEL", "LLM_PROVIDER", "TEMPERATURE",
    ):
        if os.environ.get(key):
            config[key] = os.environ[key]

    return config


def strip_json_fence(text: str) -> str:
    """LLMs like wrapping JSON in ```json fences. Remove them."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n(.*?)\n?```$", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


class LLMClient:
    """
    Thin wrapper over the Anthropic and OpenAI SDKs.

    Never raises on a missing key or a missing SDK -- check `.available` and
    read `.unavailable_reason` to find out why the LLM path is off.
    """

    def __init__(self,
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: int = 2048,
                 env_file: Optional[str] = None):
        self.config = load_env(env_file)
        self.max_tokens = max_tokens
        self._client = None
        self.available = False
        self.unavailable_reason = ""

        anthropic_key = self.config.get("ANTHROPIC_API_KEY")
        openai_key = self.config.get("OPENAI_API_KEY")
        has_anthropic = not _looks_like_placeholder(anthropic_key)
        has_openai = not _looks_like_placeholder(openai_key)

        self.provider = (
            provider
            or self.config.get("LLM_PROVIDER")
            or ("anthropic" if has_anthropic else "openai" if has_openai else "")
        ).lower().strip()

        if temperature is not None:
            self.temperature = temperature
        else:
            try:
                self.temperature = float(self.config.get("TEMPERATURE", 0.8))
            except ValueError:
                self.temperature = 0.8

        if not self.provider:
            self.unavailable_reason = (
                "No API key found. Put ANTHROPIC_API_KEY or OPENAI_API_KEY in .env "
                "(copy .env.example to .env first)."
            )
            self.model = ""
            return

        if self.provider == "anthropic":
            self.model = model or self.config.get("ANTHROPIC_MODEL") or DEFAULT_MODELS["anthropic"]
            self.api_key = anthropic_key
        elif self.provider == "openai":
            self.model = model or self.config.get("OPENAI_MODEL") or DEFAULT_MODELS["openai"]
            self.api_key = openai_key
        else:
            self.model = ""
            self.unavailable_reason = f"Unknown LLM_PROVIDER: {self.provider!r}"
            return

        if _looks_like_placeholder(self.api_key):
            self.unavailable_reason = (
                f"No usable key for provider {self.provider!r}. "
                f"Set {self.provider.upper()}_API_KEY in .env."
            )
            return

        self._init_sdk()

    def _init_sdk(self) -> None:
        try:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            else:
                from openai import OpenAI
                org = self.config.get("OPENAI_ORG_ID")
                kwargs = {"api_key": self.api_key}
                if not _looks_like_placeholder(org):
                    kwargs["organization"] = org
                self._client = OpenAI(**kwargs)
            self.available = True
        except ImportError as e:
            self.unavailable_reason = (
                f"SDK not installed for {self.provider} ({e}). "
                f"Run: pip install -r requirements.txt"
            )
        except Exception as e:
            self.unavailable_reason = f"Could not create {self.provider} client: {e}"

    def describe(self) -> str:
        if self.available:
            return f"LLM ready: {self.provider} / {self.model}"
        return f"LLM disabled: {self.unavailable_reason}"

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    def complete(self, prompt: str,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 retries: int = 3) -> str:
        """
        Send one prompt, get the text back. Returns "" if the LLM is disabled.
        Retries transient failures with exponential backoff.
        """
        if not self.available:
            return ""

        temp = self.temperature if temperature is None else temperature
        tokens = max_tokens or self.max_tokens
        last_error: Optional[Exception] = None

        for attempt in range(retries):
            try:
                if self.provider == "anthropic":
                    return self._complete_anthropic(prompt, temp, tokens)
                return self._complete_openai(prompt, temp, tokens)
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

        print(f"   [llm] call failed after {retries} attempts: {last_error}")
        return ""

    def _complete_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if any(self.model.startswith(m) for m in _NO_SAMPLING_PARAMS):
            # These models reject temperature/top_p/top_k with a 400. Map the
            # requested temperature onto the effort knob instead: low temp ->
            # tighter/cheaper reasoning, high temp -> more deliberation.
            kwargs["output_config"] = {
                "effort": "low" if temperature <= 0.7 else "medium"
            }
        else:
            kwargs["temperature"] = temperature

        response = self._client.messages.create(**kwargs)

        if getattr(response, "stop_reason", None) == "refusal":
            print("   [llm] request was declined by safety classifiers")
            return ""

        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

    def _complete_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def complete_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call the LLM and parse the reply as JSON. Returns {} on failure."""
        raw = self.complete(prompt, **kwargs)
        if not raw:
            return {}

        cleaned = strip_json_fence(raw)
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            # Model wrapped the JSON in prose -- grab the first {...} block.
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            print(f"   [llm] could not parse JSON from: {raw[:120]}...")
            return {}

    def generate_queries(self, prompt: str, **kwargs) -> List[str]:
        """
        Run a query-generation prompt (the V1/V2/V3 templates in task1_prompt.py)
        and return the list of queries.
        """
        data = self.complete_json(prompt, **kwargs)
        queries = data.get("queries", [])
        if isinstance(queries, str):
            queries = [queries]
        return [q.strip() for q in queries if isinstance(q, str) and q.strip()]


_SHARED_CLIENT: Optional[LLMClient] = None


def get_client(**kwargs) -> LLMClient:
    """Get the shared client (created once per process)."""
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None or kwargs:
        client = LLMClient(**kwargs)
        if not kwargs:
            _SHARED_CLIENT = client
        return client
    return _SHARED_CLIENT


def main():
    """`python src/llm_client.py` -- check whether the API key is wired up."""
    client = get_client()
    print("\n" + "=" * 60)
    print("  LLM configuration check")
    print("=" * 60)
    print(f"\n  {client.describe()}")

    if not client.available:
        print("\n  To enable the LLM:")
        print("    1. cp .env.example .env")
        print("    2. add ANTHROPIC_API_KEY=sk-ant-...  (or OPENAI_API_KEY=sk-...)")
        print()
        return

    print("\n  Sending a test prompt...")
    reply = client.complete("Răspunde cu un singur cuvânt: salut", max_tokens=64)
    print(f"  Reply: {reply!r}")
    print("\n  Working.\n")


if __name__ == "__main__":
    main()
