"""Alibaba Token Plan image-generation backend (Qwen-Image / Wan-Image).

Text-to-image and image editing go through the *native* synchronous DashScope endpoint on
the Token Plan host — the plan's OpenAI-compatible surface serves text models only:

    POST {base_url}/api/v1/services/aigc/multimodal-generation/generation

The prompt rides in ``input.messages[0].content[0].text``; source images (editing /
style references) are extra ``{"image": <url>}`` items in the same ``content`` array.
The answer arrives as ``output.choices[0].message.content[*].image`` — a URL that the
vendor purges after ~24 h, so it is downloaded into ``$HERMES_HOME/cache/generated/images``.

Model precedence: caller ``model=`` → ``ALIBABA_IMAGE_MODEL`` → ``image_gen.alibaba-token-plan.model``
→ ``image_gen.model`` (when it is one of our ids) → :data:`DEFAULT_MODEL`.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # shared helpers when the directory is symlinked into ~/.hermes/plugins
    sys.path.insert(0, str(_ROOT))

import hermes_tpm as tpm  # noqa: E402
from agent.image_gen_provider import (  # noqa: E402
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_url_image,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen-image-3.0-pro"

MODELS: dict[str, dict[str, Any]] = {
    "qwen-image-3.0-pro": {
        "display": "Qwen Image 3.0 Pro",
        "speed": "~10-30s",
        "strengths": "Qwen image generation + editing; best text rendering in images",
        "price": "Credits (Token Plan)",
    },
    "wan2.7-image-pro": {
        "display": "Wan 2.7 Image Pro",
        "speed": "~10-40s",
        "strengths": "Wan 2.7 Pro: 4K text-to-image (2K when an image is supplied), strong realism",
        "price": "Credits (Token Plan)",
    },
    "wan2.7-image": {
        "display": "Wan 2.7 Image",
        "speed": "~5-20s",
        "strengths": "Wan 2.7 fast tier: quicker, cheaper, up to 2K",
        "price": "Credits (Token Plan)",
    },
}

# Aspect → explicit pixel size. ``W*H`` is honored by every family in the plan; the
# ``1K``/``2K`` presets are square without an input image, so they cannot express aspect.
ASPECT_SIZE: dict[str, str] = {
    "landscape": "2048*1152",
    "square": "2048*2048",
    "portrait": "1152*2048",
}

MAX_SOURCE_IMAGES = 9  # the API accepts at most 9 ``image`` items per request
REQUEST_TIMEOUT = 180.0


def _resolve_model(caller_model: Optional[str] = None) -> str:
    """caller → env → provider config → top-level ``image_gen.model`` → default."""
    config = tpm.config_section("image_gen")
    candidates = (
        caller_model,
        os.environ.get("ALIBABA_IMAGE_MODEL"),
        config.get("model"),
        tpm.raw_section("image_gen").get("model"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return DEFAULT_MODEL


def _resolve_size(aspect_ratio: str) -> str:
    """Configured ``size`` wins (``1K``/``2K``/``4K`` preset or ``W*H``), else the aspect map."""
    configured = tpm.config_section("image_gen").get("size")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return ASPECT_SIZE[aspect_ratio]


def _image_from_response(data: dict[str, Any]) -> Optional[str]:
    choices = ((data.get("output") or {}).get("choices") or [])
    for choice in choices:
        content = ((choice or {}).get("message") or {}).get("content") or []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("image"), str) and item["image"].strip():
                return item["image"].strip()
    return None


class AlibabaTokenPlanImageGenProvider(ImageGenProvider):
    """Image generation/editing on the Alibaba Token Plan subscription."""

    name = tpm.PROVIDER_NAME
    display_name = f"{tpm.DISPLAY_NAME} (Qwen/Wan images)"

    def is_available(self) -> bool:
        return bool(tpm.api_key())

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": model_id, **meta} for model_id, meta in MODELS.items()]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": f"{tpm.DISPLAY_NAME} (images)",
            "badge": "subscription",
            "tag": "Qwen Image 3.0 Pro / Wan 2.7 — text-to-image and image editing on an Alibaba "
                   "Model Studio Token Plan (sk-sp- key, native DashScope endpoint)",
            "env_vars": [
                {
                    "key": tpm.KEY_ENV,
                    "prompt": "Alibaba Token Plan API key (sk-sp-…), used for image/video only",
                    "url": tpm.CONSOLE_KEYS_URL,
                }
            ],
        }

    def capabilities(self) -> dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": MAX_SOURCE_IMAGES - 1,
                "max_source_images": MAX_SOURCE_IMAGES}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model_id = _resolve_model(kwargs.get("model"))

        def fail(error: str, error_type: str) -> dict[str, Any]:
            return error_response(error=error, error_type=error_type, provider=self.name,
                                  model=model_id, prompt=prompt, aspect_ratio=aspect)

        key = tpm.api_key()
        if not key:
            return fail(tpm.missing_key_message(), "missing_credentials")
        if not prompt:
            return fail("Prompt is required", "invalid_input")

        sources: list[str] = [image_url] if isinstance(image_url, str) and image_url.strip() else []
        sources.extend(normalize_reference_images(reference_image_urls) or [])
        if len(sources) > MAX_SOURCE_IMAGES:
            return fail(f"This backend accepts at most {MAX_SOURCE_IMAGES} source images "
                        f"(got {len(sources)})", "too_many_references")

        content: list[dict[str, Any]] = [{"text": prompt}]
        for source in sources:
            try:
                content.append({"image": tpm.media_url(source)})
            except (OSError, ValueError) as exc:
                return fail(f"Could not use source image {source!r}: {exc}", "invalid_image")

        payload: dict[str, Any] = {
            "model": model_id,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {
                "size": _resolve_size(aspect),
                "n": max(1, min(4, tpm.option("image_gen", "n", 1))),
                "watermark": tpm.option("image_gen", "watermark", False),
                "prompt_extend": tpm.option("image_gen", "prompt_extend", True),
            },
        }
        timeout = float(tpm.option("image_gen", "request_timeout", REQUEST_TIMEOUT))

        url = f"{tpm.base_url('image_gen')}{tpm.IMAGE_PATH}"
        data, error = tpm.request_json("POST", url, key, payload=payload, timeout=timeout)
        if error:
            logger.warning("Alibaba image generation failed (model=%s): %s", model_id, error)
            return fail(f"Alibaba Token Plan image generation failed: {error}", "api_error")
        assert data is not None

        remote = _image_from_response(data)
        if not remote:
            return fail("Alibaba returned no image URL in the response", "empty_response")

        try:  # URLs expire after ~24 h — keep a local copy when possible
            image_ref = str(save_url_image(remote, prefix=f"alibaba_{model_id}"))
        except Exception as exc:
            logger.debug("Could not cache the generated image (%s); returning the remote URL", exc)
            image_ref = remote

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return {
            "success": True,
            "image": image_ref,
            "model": model_id,
            "prompt": prompt,
            "aspect_ratio": aspect,
            "modality": "image" if sources else "text",
            "provider": self.name,
            **tpm.merge_extra(
                {"size": payload["parameters"]["size"], "request_id": data.get("request_id")},
                {"width": usage.get("width"), "height": usage.get("height"),
                 "image_count": usage.get("image_count")} if usage else None,
                {} if image_ref == remote else {"source_url": remote},
            ),
        }


def register(ctx: Any) -> None:
    """Plugin entry point — wire the provider into the image-gen registry."""
    ctx.register_image_gen_provider(AlibabaTokenPlanImageGenProvider())
