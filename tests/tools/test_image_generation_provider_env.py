from __future__ import annotations

import json

import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


@pytest.fixture
def image_tool():
    import importlib
    import tools.image_generation_tool as mod

    return importlib.reload(mod)


class _FakeVertexProvider(ImageGenProvider):
    def __init__(self, calls):
        self.calls = calls

    @property
    def name(self) -> str:
        return "vertex"

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                **kwargs,
            }
        )
        return {
            "success": True,
            "image": "/tmp/vertex-test.png",
            "provider": "vertex",
            "model": kwargs.get("model"),
        }


def _clear_image_env(monkeypatch):
    for key in (
        "HERMES_IMAGE_GEN_PROVIDER",
        "IMAGE_GEN_PROVIDER",
        "IMAGE_PROVIDER",
        "HERMES_IMAGE_GEN_MODEL",
        "VERTEX_IMAGE_MODEL",
        "GEMINI_IMAGE_MODEL",
        "IMAGE_MODEL",
        "FAL_IMAGE_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_no_env_no_config_returns_none(image_tool, monkeypatch, tmp_path):
    _clear_image_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    assert image_tool._read_configured_image_provider() is None


def test_image_provider_vertex_dispatches_to_plugin_not_fal(image_tool, monkeypatch, tmp_path):
    from agent import image_gen_registry as registry_module
    from hermes_cli import plugins as plugins_module

    _clear_image_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("IMAGE_PROVIDER", "vertex")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    calls = []
    provider = _FakeVertexProvider(calls)
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
    monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "vertex" else None)
    monkeypatch.setattr(image_tool, "image_generate_tool", lambda **kwargs: pytest.fail("FAL fallback should not run"))

    dispatched = image_tool._handle_image_generate({"prompt": "draw cat", "aspect_ratio": "square"})
    payload = json.loads(dispatched)

    assert payload["success"] is True
    assert payload["provider"] == "vertex"
    assert calls == [{"prompt": "draw cat", "aspect_ratio": "square"}]


def test_hermes_image_gen_provider_alias_dispatches(image_tool, monkeypatch, tmp_path):
    from agent import image_gen_registry as registry_module
    from hermes_cli import plugins as plugins_module

    _clear_image_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IMAGE_GEN_PROVIDER", "vertex_gemini")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    calls = []
    provider = _FakeVertexProvider(calls)
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
    monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "vertex_gemini" else None)

    dispatched = image_tool._dispatch_to_plugin_provider("draw alias", "landscape")
    payload = json.loads(dispatched)

    assert payload["success"] is True
    assert payload["provider"] == "vertex"
    assert calls[0]["prompt"] == "draw alias"


def test_unknown_env_provider_returns_structured_error(image_tool, monkeypatch, tmp_path):
    from hermes_cli import plugins as plugins_module

    _clear_image_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("IMAGE_PROVIDER", "missing-image-backend")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
    dispatched = image_tool._dispatch_to_plugin_provider("draw cat", "landscape")
    payload = json.loads(dispatched)

    assert payload["success"] is False
    assert payload["error_type"] == "provider_not_registered"
    assert "image_gen.provider='missing-image-backend'" in payload["error"]


def test_vertex_image_model_env_passed_to_provider_dispatch(image_tool, monkeypatch, tmp_path):
    from agent import image_gen_registry as registry_module
    from hermes_cli import plugins as plugins_module

    _clear_image_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("IMAGE_PROVIDER", "vertex")
    monkeypatch.setenv("VERTEX_IMAGE_MODEL", "gemini-2.5-flash-image")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    calls = []
    provider = _FakeVertexProvider(calls)
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda force=False: None)
    monkeypatch.setattr(registry_module, "get_provider", lambda name: provider if name == "vertex" else None)

    dispatched = image_tool._dispatch_to_plugin_provider("draw model", "portrait")
    payload = json.loads(dispatched)

    assert payload["success"] is True
    assert calls[0]["model"] == "gemini-2.5-flash-image"


def test_fal_fallback_unchanged_when_provider_unset(image_tool, monkeypatch, tmp_path):
    _clear_image_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    assert image_tool._dispatch_to_plugin_provider("draw cat", "landscape") is None
