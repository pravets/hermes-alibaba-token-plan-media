"""Test bootstrap.

The plugins talk to Hermes core (``agent.*`` / ``hermes_cli.*``). Those modules exist when
the tests run inside a Hermes install, but not on a bare CI runner — so when the real import
fails we install minimal stand-ins that preserve the exact contract the plugins rely on.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_core_stubs() -> None:
    try:  # real Hermes install present
        import agent.image_gen_provider  # noqa: F401
        import agent.video_gen_provider  # noqa: F401

        return
    except Exception:
        pass

    agent_pkg = _module("agent")
    agent_pkg.__path__ = []
    _module("agent.provider_base").CatalogProviderBase = type("CatalogProviderBase", (), {"default_model": lambda self: None})

    image = _module("agent.image_gen_provider")

    class ImageGenProvider:
        def capabilities(self):
            return {"modalities": ["text"], "max_reference_images": 0}

    image.ImageGenProvider = ImageGenProvider
    image.VALID_ASPECT_RATIOS = ("landscape", "square", "portrait")
    image.DEFAULT_ASPECT_RATIO = "landscape"

    def resolve_aspect_ratio(value):
        v = value.strip().lower() if isinstance(value, str) else ""
        return v if v in image.VALID_ASPECT_RATIOS else image.DEFAULT_ASPECT_RATIO

    def normalize_reference_images(value):
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return None
        return [i.strip() for i in value if isinstance(i, str) and i.strip()] or None

    def success_response(**kw):
        payload = {"success": True, "image": kw.get("image"), "model": kw.get("model"),
                   "prompt": kw.get("prompt"), "aspect_ratio": kw.get("aspect_ratio"),
                   "modality": kw.get("modality", "text"), "provider": kw.get("provider")}
        payload.update({k: v for k, v in (kw.get("extra") or {}).items() if k not in payload})
        return payload

    def error_response(**kw):
        return {"success": False, "image": None, "error": kw.get("error"),
                "error_type": kw.get("error_type", "provider_error"), "model": kw.get("model", ""),
                "prompt": kw.get("prompt", ""), "aspect_ratio": kw.get("aspect_ratio", "landscape"),
                "provider": kw.get("provider", "")}

    def save_url_image(url, *, prefix="image", **_):
        path = Path(os.environ.get("TPM_TEST_CACHE", "/tmp/tpm-cache")) / f"{prefix}-{abs(hash(url))}.png"
        return path

    image.resolve_aspect_ratio = resolve_aspect_ratio
    image.normalize_reference_images = normalize_reference_images
    image.success_response = success_response
    image.error_response = error_response
    image.save_url_image = save_url_image

    video = _module("agent.video_gen_provider")

    class VideoGenProvider:
        def capabilities(self):
            return {"modalities": ["text"]}

    video.VideoGenProvider = VideoGenProvider
    video.DEFAULT_ASPECT_RATIO = "16:9"
    video.DEFAULT_RESOLUTION = "720p"

    def success_response_video(**kw):
        return {"success": True, "video": kw.get("video"), "model": kw.get("model"),
                "prompt": kw.get("prompt"), "modality": kw.get("modality", "text"),
                "aspect_ratio": kw.get("aspect_ratio", ""), "duration": int(kw.get("duration") or 0),
                "provider": kw.get("provider"),
                **{k: v for k, v in (kw.get("extra") or {}).items()}}

    def error_response_video(**kw):
        return {"success": False, "video": None, "error": kw.get("error"),
                "error_type": kw.get("error_type", "provider_error"), "model": kw.get("model", ""),
                "prompt": kw.get("prompt", ""), "aspect_ratio": kw.get("aspect_ratio", ""),
                "provider": kw.get("provider", "")}

    def save_url_video(url, *, prefix="video", **_):
        return Path(os.environ.get("TPM_TEST_CACHE", "/tmp/tpm-cache")) / f"{prefix}-{abs(hash(url))}.mp4"

    video.success_response = success_response_video
    video.error_response = error_response_video
    video.save_url_video = save_url_video

    secret_scope = _module("agent.secret_scope")
    secret_scope.get_secret_str = lambda name: os.environ.get(name, "")

    hermes_cli = _module("hermes_cli")
    hermes_cli.__path__ = []
    _module("hermes_cli.config").load_config = lambda: {}


_install_core_stubs()
