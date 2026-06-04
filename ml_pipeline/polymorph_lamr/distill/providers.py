"""Provider routing for distillation teachers.

Three providers are wired for the open-weight teacher ensemble (E3):

* **AWS Bedrock** — litellm-native (`bedrock/<model>`), routed through the Bedrock
  Converse API. Credentials come from the standard AWS chain (env / profile / SSO
  login provider — the last needs ``botocore[crt]``). No API key env; the region
  is resolved from ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` (DeepSeek V3.2 is
  ON_DEMAND in us-east-1, us-west-2, eu-north-1). This is the **primary teacher**
  (`deepseek.v3.2`): strong, cheap (~17x cheaper than the old Vercel path), no
  reasoning-token overhead.
* **OpenRouter** — litellm-native (`openrouter/<vendor>/<model>`); key from
  ``OPENROUTER_API_KEY``. Free open-weight teachers (e.g. Kimi). Free models share
  an upstream pool and are frequently rate-limited (HTTP 429); the ensemble
  tolerates this — a teacher that errors is dropped from per-chunk best-QC
  selection. Kept as a zero-cost opportunistic secondary.
* **Vercel AI Gateway** — an OpenAI-compatible endpoint (`vercel/<model>`), key
  from ``VERCEL_AI_GATEWAY_KEY``. No longer a default teacher (superseded by
  Bedrock) but retained as a routable option. STRICT RULE: the ONLY model
  permitted through the Vercel gateway is ``alibaba/qwen3.7-max`` — any other is a
  hard error at teacher-construction time, so a typo can never silently route or
  bill a disallowed gateway model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# AWS Bedrock: litellm-native; AWS credential chain, no API key.
BEDROCK_REGION_ENVS = ("AWS_REGION", "AWS_DEFAULT_REGION")

# Vercel AI Gateway: OpenAI-compatible chat-completions endpoint.
VERCEL_AI_GATEWAY_BASE = "https://ai-gateway.vercel.sh/v1"
VERCEL_AI_GATEWAY_KEY_ENV = "VERCEL_AI_GATEWAY_KEY"
# The ONLY model allowed through the Vercel AI Gateway (project rule).
VERCEL_ALLOWED_MODEL = "alibaba/qwen3.7-max"

OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

# Spec-string prefixes the CLI/config accept.
_BEDROCK_PREFIX = "bedrock/"
_VERCEL_PREFIX = "vercel/"
_OPENROUTER_PREFIX = "openrouter/"


def bedrock_region() -> str | None:
    """Resolve the Bedrock region from the AWS env chain (None lets boto3 fall
    back to the active profile's default region)."""
    for env in BEDROCK_REGION_ENVS:
        val = os.environ.get(env)
        if val:
            return val
    return None


@dataclass(frozen=True)
class Routing:
    """How litellm should reach a teacher.

    ``model`` is the litellm model id actually sent. ``api_base`` is set only for
    OpenAI-compatible custom endpoints (the Vercel gateway). ``api_key_env`` names
    the environment variable holding the credential, resolved at call time.
    ``aws_region`` is set only for Bedrock (which uses the AWS credential chain,
    not an API key).
    """

    model: str
    api_base: str | None
    api_key_env: str | None
    aws_region: str | None = None


def resolve_routing(spec_model: str) -> Routing:
    """Map a teacher spec-string to litellm routing.

    Recognized prefixes:

    * ``bedrock/<model>``  -> AWS Bedrock (litellm-native Converse). No API key —
      AWS credential chain; region from ``AWS_REGION``/``AWS_DEFAULT_REGION``.
    * ``vercel/<model>``   -> Vercel AI Gateway (OpenAI-compatible); STRICT-guarded
      to ``alibaba/qwen3.7-max`` only. The litellm id becomes ``openai/<model>``
      so litellm hits the custom ``api_base`` instead of api.openai.com.
    * ``openrouter/<...>`` -> OpenRouter (litellm-native), key ``OPENROUTER_API_KEY``.
    * anything else        -> passed through unchanged; litellm uses its default
      provider/credential resolution.
    """
    if spec_model.startswith(_BEDROCK_PREFIX):
        return Routing(
            model=spec_model,  # litellm-native, e.g. "bedrock/deepseek.v3.2"
            api_base=None,
            api_key_env=None,
            aws_region=bedrock_region(),
        )
    if spec_model.startswith(_VERCEL_PREFIX):
        model = spec_model[len(_VERCEL_PREFIX):]
        if model != VERCEL_ALLOWED_MODEL:
            raise ValueError(
                f"Vercel AI Gateway is restricted to '{VERCEL_ALLOWED_MODEL}'; "
                f"refusing to route '{model}'. (project strict rule)"
            )
        return Routing(
            model=f"openai/{model}",
            api_base=VERCEL_AI_GATEWAY_BASE,
            api_key_env=VERCEL_AI_GATEWAY_KEY_ENV,
        )
    if spec_model.startswith(_OPENROUTER_PREFIX):
        return Routing(model=spec_model, api_base=None, api_key_env=OPENROUTER_KEY_ENV)
    return Routing(model=spec_model, api_base=None, api_key_env=None)


# Default open-weight teacher ensemble. deepseek-v32 (Bedrock) is the reliable,
# cheap workhorse; kimi (OpenRouter free) contributes opportunistically at zero
# cost when the free pool has capacity. Order is not significant — best-QC
# selection is per chunk.
DEFAULT_TEACHER_SPECS: list[tuple[str, str]] = [
    ("deepseek-v32", "bedrock/deepseek.v3.2"),
    ("kimi", "openrouter/moonshotai/kimi-k2.6:free"),
]
