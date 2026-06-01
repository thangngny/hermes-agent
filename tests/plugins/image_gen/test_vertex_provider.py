from __future__ import annotations

import importlib.util
from pathlib import Path


PLUGIN_PATH = Path(__file__).resolve().parents[3] / "plugins/image_gen/vertex/__init__.py"


def _load_vertex_module():
    spec = importlib.util.spec_from_file_location("vertex_image_gen_provider_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_vertex_provider_missing_project_returns_structured_error(monkeypatch):
    module = _load_vertex_module()
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("VERTEX_AI_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_PROJECT_ID", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")

    provider = module.VertexGeminiImageProvider()
    result = provider.generate("safe abstract moodboard")

    assert result["success"] is False
    assert result["provider"] == "vertex"
    assert result["error_type"] == "missing_project"
    assert "GOOGLE_CLOUD_PROJECT" in result["error"]


def test_vertex_provider_registers_aliases():
    module = _load_vertex_module()
    registered = []

    class Ctx:
        def register_image_gen_provider(self, provider):
            registered.append(provider.name)

    module.register(Ctx())

    assert "vertex" in registered
    assert "vertex_gemini" in registered
    assert "gemini_image" in registered
    assert "google_vertex" in registered
