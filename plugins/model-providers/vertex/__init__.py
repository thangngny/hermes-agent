"""Google Vertex AI provider profile.

Implements provider id `vertex` and reuses Gemini native transport.
Auth is ADC / OAuth bearer token at runtime (no API key required).
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class VertexProfile(ProviderProfile):
    """Vertex provider with Gemini thinking_config translation."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        from agent.transports.chat_completions import _build_gemini_thinking_config

        model = context.get("model") or ""
        reasoning_config = context.get("reasoning_config")
        raw_thinking_config = _build_gemini_thinking_config(model, reasoning_config)
        if not raw_thinking_config:
            return {}
        return {"thinking_config": raw_thinking_config}


vertex = VertexProfile(
    name="vertex",
    aliases=("google-vertex", "vertex-ai", "google-vertex-ai"),
    api_mode="chat_completions",
    env_vars=(
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "VERTEX_AI_PROJECT",
        "VERTEX_AI_LOCATION",
        "GOOGLE_PROJECT_ID",
        "GOOGLE_LOCATION",
    ),
    # Marker URL interpreted by GeminiNativeClient as Vertex mode.
    base_url="vertex://google",
    auth_type="api_key",
    default_aux_model="gemini-2.5-flash",
    fallback_models=(
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    ),
)

register_provider(vertex)
