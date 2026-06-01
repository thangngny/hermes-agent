"""Vertex Gemini image generation backend for Hermes image_gen.

This provider routes Hermes' image_generate tool to Google Gen AI SDK in
Vertex mode. It is intentionally opt-in via ``image_gen.provider: vertex``;
FAL and other existing image providers keep their current behavior when the
provider is unset.
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash-image"
_PROVIDER_ALIASES = ("vertex", "vertex_gemini", "gemini_image", "google_vertex")
_SECRET_MARKERS = (".env", "api_key", "apikey", "token", "secret", "private key")


def _load_image_gen_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_project_location() -> Tuple[str, str]:
    cfg = _load_image_gen_config()
    project = (
        str(cfg.get("project") or "").strip()
        or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        or os.environ.get("VERTEX_AI_PROJECT", "").strip()
        or os.environ.get("GOOGLE_PROJECT_ID", "").strip()
    )
    location = (
        str(cfg.get("location") or "").strip()
        or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
        or os.environ.get("VERTEX_AI_LOCATION", "").strip()
        or os.environ.get("GOOGLE_LOCATION", "").strip()
        or "global"
    )
    return project, location


def _resolve_model(explicit: Optional[str] = None) -> str:
    cfg = _load_image_gen_config()
    value = (
        str(explicit or "").strip()
        or os.environ.get("VERTEX_IMAGE_MODEL", "").strip()
        or os.environ.get("GEMINI_IMAGE_MODEL", "").strip()
        or os.environ.get("IMAGE_MODEL", "").strip()
        or str(cfg.get("model") or "").strip()
        or DEFAULT_MODEL
    )
    return value


def _vertex_env_enabled() -> bool:
    raw = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _safe_prompt(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return not any(marker in lower for marker in _SECRET_MARKERS)


def _output_dir() -> Path:
    override = os.environ.get("HERMES_IMAGE_OUTPUT_DIR", "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "cache" / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_image(data: bytes, *, prefix: str = "vertex_gemini", extension: str = "png") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _output_dir() / f"{prefix}_{ts}.{extension}"
    path.write_bytes(data)
    return path


def _maybe_bytes(value: Any) -> Optional[bytes]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        try:
            return base64.b64decode(value)
        except Exception:  # noqa: BLE001
            return None
    return None


def _extract_image_bytes(response: Any) -> Optional[Tuple[bytes, str]]:
    """Best-effort extraction for google.genai GenerateContentResponse shapes."""
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
            if inline is not None:
                data = _maybe_bytes(getattr(inline, "data", None))
                mime = str(getattr(inline, "mime_type", "") or getattr(inline, "mimeType", "") or "image/png")
                if data:
                    ext = "jpg" if "jpeg" in mime else "webp" if "webp" in mime else "png"
                    return data, ext
            try:
                image = part.as_image()
                if image is not None:
                    import io

                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    return buf.getvalue(), "png"
            except Exception:  # noqa: BLE001
                pass
    return None


class VertexGeminiImageProvider(ImageGenProvider):
    def __init__(self, name: str = "vertex") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return "Vertex Gemini Image"

    def is_available(self) -> bool:
        project, _location = _resolve_project_location()
        try:
            import google.genai  # noqa: F401
        except Exception:
            return False
        return bool(project and _vertex_env_enabled())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "Gemini 2.5 Flash Image",
                "speed": "variable",
                "strengths": "Vertex Gemini image generation/editing via Google Gen AI SDK",
                "price": "Vertex billing",
            },
            {
                "id": "gemini-3-pro-image-preview",
                "display": "Gemini 3 Pro Image Preview",
                "speed": "variable",
                "strengths": "Preview image model when enabled for the project",
                "price": "Vertex billing",
            },
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Vertex Gemini Image",
            "badge": "gcp",
            "tag": "Gemini image generation through Vertex AI / Google Gen AI SDK.",
            "env_vars": [
                {"key": "GOOGLE_CLOUD_PROJECT", "prompt": "Google Cloud project", "url": "https://console.cloud.google.com/"},
                {"key": "GOOGLE_CLOUD_LOCATION", "prompt": "Vertex location, e.g. global or us-central1", "url": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations"},
                {"key": "GOOGLE_GENAI_USE_VERTEXAI", "prompt": "Set to true for Vertex mode", "url": "https://googleapis.github.io/python-genai/"},
            ],
        }

    def generate(self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, **kwargs: Any) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model = _resolve_model(str(kwargs.get("model") or "").strip() or None)
        project, location = _resolve_project_location()

        if not prompt:
            return error_response(error="Prompt is required", error_type="invalid_argument", provider=self.name, model=model, aspect_ratio=aspect)
        if not _safe_prompt(prompt):
            return error_response(error="Prompt appears to request or contain secrets; refusing image generation.", error_type="secret_guard", provider=self.name, model=model, prompt="[redacted]", aspect_ratio=aspect)
        if not project:
            return error_response(error="GOOGLE_CLOUD_PROJECT or image_gen.project is required for Vertex image generation.", error_type="missing_project", provider=self.name, model=model, prompt=prompt, aspect_ratio=aspect)
        if not _vertex_env_enabled():
            return error_response(error="GOOGLE_GENAI_USE_VERTEXAI=true is required for Vertex image generation.", error_type="vertex_mode_required", provider=self.name, model=model, prompt=prompt, aspect_ratio=aspect)

        try:
            from google import genai
            from google.genai import types
        except Exception as exc:  # noqa: BLE001
            return error_response(error=f"google-genai dependency is unavailable: {type(exc).__name__}", error_type="missing_dependency", provider=self.name, model=model, prompt=prompt, aspect_ratio=aspect)

        try:
            client = genai.Client(vertexai=True, project=project, location=location)
            config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            extracted = _extract_image_bytes(response)
            if not extracted:
                return error_response(error="Vertex Gemini response did not contain an image artifact.", error_type="no_image_artifact", provider=self.name, model=model, prompt=prompt, aspect_ratio=aspect)
            data, ext = extracted
            image_path = _write_image(data, extension=ext)
            return success_response(
                image=str(image_path),
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
                provider=self.name,
                extra={"uses_vertex": True, "project_configured": True, "location": location},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vertex Gemini image generation failed: %s", exc)
            return error_response(error=f"Vertex Gemini image generation failed: {type(exc).__name__}: {exc}", error_type="api_error", provider=self.name, model=model, prompt=prompt, aspect_ratio=aspect)


def register(ctx) -> None:
    for name in _PROVIDER_ALIASES:
        ctx.register_image_gen_provider(VertexGeminiImageProvider(name=name))
