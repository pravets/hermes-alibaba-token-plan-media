"""Alibaba Token Plan video-generation backend (Wan 3.0 / HappyHorse 1.1).

Every model on the plan is an **asynchronous** job on the native DashScope paths:

    POST {base_url}/api/v1/services/aigc/video-generation/video-synthesis   (+ X-DashScope-Async: enable)
    GET  {base_url}/api/v1/tasks/{task_id}

Routing follows the Hermes convention: ``image_url`` present ⇒ image-to-video, absent ⇒
text-to-video. Which endpoint semantics apply is decided by the configured *family* — the
agent never picks a model. Source images ride in ``input.media`` as
``{"type": "first_frame" | "last_frame" | "reference_image", "url": …}`` (public URL or
``data:`` URI, so local files work). ``video_url`` from a finished task lives ~24 h, so the
mp4 is downloaded into ``$HERMES_HOME/cache/generated/videos`` and the local path returned.

Family precedence: ``model=`` (tool layer) → ``ALIBABA_VIDEO_MODEL`` →
``video_gen.alibaba-token-plan.model`` → ``video_gen.model`` → :data:`DEFAULT_MODEL`.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # shared helpers when the directory is symlinked into ~/.hermes/plugins
    sys.path.insert(0, str(_ROOT))

import hermes_tpm as tpm  # noqa: E402
from agent.video_gen_provider import (  # noqa: E402
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    save_url_video,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "wan3.0-video"

_SHORT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")

# ``image_mode``: how a source image is submitted —
#   "first_frame"  → media[{"type": "first_frame"}]
#   "reference"    → media[{"type": "reference_image"}] (character/style references)
# ``audio`` / ``seed``: whether the family's schema has the knob at all.
FAMILIES: dict[str, dict[str, Any]] = {
    "wan3.0-video": {
        "display": "Wan 3.0 Video",
        "speed": "~1-5min",
        "price": "Credits (Token Plan)",
        "strengths": "All-in-one: text, first-frame, first+last frame, reference images; native audio, "
                     "2-30 s, smart duration, adaptive ratio",
        "modalities": ("text", "image"),
        "durations": (2, 30),
        "resolutions": ("480P", "720P", "1080P"),
        "ratios": _SHORT_RATIOS,
        "audio": True,
        "seed": True,
        "image_mode": "first_frame",
        "smart_duration": True,
        "adaptive_ratio": True,
    },
    "wan3.0-video-prime": {
        "display": "Wan 3.0 Video Prime",
        "speed": "~30s-3min",
        "price": "Credits (Token Plan)",
        "strengths": "Speed-optimised twin of Wan 3.0 Video; same capabilities, noticeably faster",
        "modalities": ("text", "image"),
        "durations": (2, 30),
        "resolutions": ("480P", "720P", "1080P"),
        "ratios": _SHORT_RATIOS,
        "audio": True,
        "seed": True,
        "image_mode": "first_frame",
        "smart_duration": True,
        "adaptive_ratio": True,
    },
    "happyhorse-1.1-t2v": {
        "display": "HappyHorse 1.1 (text-to-video)",
        "speed": "~1-4min",
        "price": "Credits (Token Plan)",
        "strengths": "Cinematic text-to-video with native audio, 3-15 s, up to 1080P",
        "modalities": ("text",),
        "durations": (3, 15),
        "resolutions": ("480P", "720P", "1080P"),
        "ratios": ("16:9", "9:16", "1:1", "4:3", "3:4"),
        "audio": False,  # the Token Plan video doc exposes no audio toggle for HappyHorse
        "seed": False,
        "image_mode": None,
    },
    "happyhorse-1.1-i2v": {
        "display": "HappyHorse 1.1 (first-frame to video)",
        "speed": "~1-4min",
        "price": "Credits (Token Plan)",
        "strengths": "Animates exactly one first-frame image; aspect follows the image",
        "modalities": ("image",),
        "durations": (3, 15),
        "resolutions": ("480P", "720P", "1080P"),
        "ratios": (),  # i2v rejects parameters.ratio
        "audio": False,
        "seed": False,
        "image_mode": "first_frame",
    },
    "happyhorse-1.1-r2v": {
        "display": "HappyHorse 1.1 (reference to video)",
        "speed": "~1-4min",
        "price": "Credits (Token Plan)",
        "strengths": "Keeps a character consistent across scenes from reference images",
        "modalities": ("image",),
        "durations": (3, 15),
        "resolutions": ("480P", "720P", "1080P"),
        "ratios": ("16:9", "9:16", "1:1"),
        "audio": False,
        "seed": False,
        "image_mode": "reference",
    },
}

# Tool resolution vocabulary → the API's upper-case enum.
_RESOLUTION_ALIASES = {
    "360p": "480P", "480p": "480P", "540p": "480P", "720p": "720P", "768p": "720P",
    "1080p": "1080P", "1440p": "1080P", "2k": "1080P", "2160p": "1080P", "4k": "1080P",
}

TERMINAL_FAILURES = {"FAILED", "CANCELED", "UNKNOWN"}
MAX_REFERENCE_IMAGES = 10


def _resolve_family(explicit: Optional[str]) -> str:
    config = tpm.config_section("video_gen")
    candidates = (
        explicit,
        os.environ.get("ALIBABA_VIDEO_MODEL"),
        config.get("model"),
        tpm.config_section("video_gen").get("model"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip() in FAMILIES:
            return candidate.strip()
    return DEFAULT_MODEL


def _resolve_resolution(family: dict[str, Any], resolution: str) -> Optional[str]:
    value = _RESOLUTION_ALIASES.get((resolution or "").strip().lower())
    if value is None and isinstance(resolution, str) and resolution.strip():
        value = resolution.strip().upper()
    return value if value in family["resolutions"] else None


def _clamp_duration(family: dict[str, Any], duration: Optional[int]) -> Optional[int]:
    if duration is None:
        return None
    try:
        seconds = int(duration)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:  # smart duration (-1); only Wan 3.0 documents it
        return -1 if family.get("smart_duration") else None
    low, high = family["durations"]
    return max(low, min(high, seconds))


class AlibabaTokenPlanVideoGenProvider(VideoGenProvider):
    """Text-to-video and image-to-video on the Alibaba Token Plan subscription."""

    name = tpm.PROVIDER_NAME
    display_name = f"{tpm.DISPLAY_NAME} (Wan/HappyHorse video)"

    def is_available(self) -> bool:
        return bool(tpm.api_key())

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": family_id,
                "display": family["display"],
                "speed": family["speed"],
                "strengths": family["strengths"],
                "price": family["price"],
                "modalities": list(family["modalities"]),
                "min_duration": family["durations"][0],
                "max_duration": family["durations"][1],
            }
            for family_id, family in FAMILIES.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": f"{tpm.DISPLAY_NAME} (video)",
            "badge": "subscription",
            "tag": "Wan 3.0 Video and HappyHorse 1.1 — text-to-video and image-to-video on an Alibaba "
                   "Model Studio Token Plan (sk-sp- key, async task API)",
            "env_vars": [
                {
                    "key": tpm.KEY_ENV,
                    "prompt": "Alibaba Token Plan API key (sk-sp-…), used for image/video only",
                    "url": tpm.CONSOLE_KEYS_URL,
                }
            ],
        }

    def capabilities(self) -> dict[str, Any]:
        durations = [family["durations"] for family in FAMILIES.values()]
        ratios: list[str] = []
        resolutions: list[str] = []
        for family in FAMILIES.values():
            ratios.extend(r for r in family["ratios"] if r not in ratios)
            resolutions.extend(r for r in family["resolutions"] if r not in resolutions)
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ratios,
            "resolutions": sorted(resolutions),
            "min_duration": min(low for low, _ in durations),
            "max_duration": max(high for _, high in durations),
            "supports_audio": True,
            "supports_negative_prompt": False,  # the Token Plan video API has no such field
            "supports_seed": True,
            "supports_upscale": False,
            "max_reference_images": MAX_REFERENCE_IMAGES,
        }

    # ── internals ──────────────────────────────────────────────────────────────

    def _poll(self, task_id: str, key: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        interval = max(2.0, float(tpm.option("video_gen", "poll_interval", 8.0)))
        deadline = time.monotonic() + max(60.0, float(tpm.option("video_gen", "poll_timeout", 900.0)))
        url = f"{tpm.base_url('video_gen')}{tpm.TASK_PATH.format(task_id=task_id)}"
        last_status = "PENDING"
        while True:
            data, error = tpm.request_json("GET", url, key, timeout=60.0)
            if error:
                return None, f"task {task_id} poll failed: {error}"
            output = (data or {}).get("output") or {}
            last_status = str(output.get("task_status") or "UNKNOWN")
            if last_status == "SUCCEEDED":
                return output, None
            if last_status in TERMINAL_FAILURES:
                return None, (f"task {task_id} ended with {last_status}: "
                              f"{output.get('code') or 'error'}: {output.get('message') or 'no message'}")
            if time.monotonic() >= deadline:
                return None, f"task {task_id} did not finish in time (last status {last_status})"
            time.sleep(interval)

    # ── generate ───────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[list[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        family_id = _resolve_family(model)
        family = FAMILIES[family_id]

        def fail(error: str, error_type: str) -> dict[str, Any]:
            return error_response(error=error, error_type=error_type, provider=self.name,
                                  model=family_id, prompt=prompt, aspect_ratio=aspect_ratio or "")

        key = tpm.api_key()
        if not key:
            return fail(tpm.missing_key_message(), "missing_credentials")
        if not prompt:
            return fail("Prompt is required", "invalid_input")

        sources: list[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources = [image_url.strip()]
        elif reference_image_urls:
            sources = [str(item).strip() for item in reference_image_urls if str(item).strip()]
            sources = sources[:MAX_REFERENCE_IMAGES]

        modality = "image" if sources else "text"
        if sources and modality not in family["modalities"]:
            return fail(f"{family['display']} cannot take a source image. Pick a family with image input "
                        f"(`hermes tools` → Video Generation) or drop image_url.", "modality_unsupported")
        if not sources and "text" not in family["modalities"]:
            return fail(f"{family['display']} needs a source image (`image_url`).", "missing_image")

        media: list[dict[str, str]] = []
        media_type = ""
        if sources:
            # ``image_url`` = the frame to animate; explicit ``reference_image_urls`` are character/style refs.
            media_type = "first_frame" if sources[0] == (image_url or "").strip() else "reference_image"
            if media_type == "first_frame" and family["image_mode"] == "reference":
                media_type = "reference_image"
            try:
                media = [{"type": media_type, "url": tpm.media_url(source)} for source in sources]
            except (OSError, ValueError) as exc:
                return fail(f"Could not use source image: {exc}", "invalid_image")

        parameters: dict[str, Any] = {
            "prompt_extend": tpm.option("video_gen", "prompt_extend", True),
            "watermark": tpm.option("video_gen", "watermark", False),
        }
        resolved_resolution = _resolve_resolution(family, resolution)
        if resolved_resolution:
            parameters["resolution"] = resolved_resolution
        if media_type == "first_frame" and family.get("adaptive_ratio"):
            # First-frame jobs follow the still image's shape; a fixed ratio would stretch it.
            parameters["ratio"] = "adaptive"
        elif aspect_ratio and aspect_ratio in family["ratios"]:
            parameters["ratio"] = aspect_ratio
        elif family["ratios"] and aspect_ratio in ("", None):
            parameters["ratio"] = "16:9"
        clamped = _clamp_duration(family, duration)
        if clamped is not None:
            parameters["duration"] = clamped
        if family["audio"] and audio is not None:
            parameters["audio"] = bool(audio)
        if family["seed"] and seed is not None:
            parameters["seed"] = int(seed)

        video_input: dict[str, Any] = {"prompt": prompt}
        if media:
            video_input["media"] = media
        payload = {"model": family_id, "input": video_input, "parameters": parameters}

        base = tpm.base_url("video_gen")
        submitted, error = tpm.request_json(
            "POST", f"{base}{tpm.VIDEO_SUBMIT_PATH}", key, payload=payload,
            headers={"X-DashScope-Async": "enable"},
            timeout=float(tpm.option("video_gen", "submit_timeout", 120.0)),
        )
        if error:
            logger.warning("Alibaba video submit failed (family=%s): %s", family_id, error)
            return fail(f"Alibaba Token Plan video submit failed: {error}", "api_error")

        task_id = str(((submitted or {}).get("output") or {}).get("task_id") or "").strip()
        if not task_id:
            return fail("Alibaba returned no task_id for the video job", "empty_response")

        output, error = self._poll(task_id, key)
        if error:
            logger.warning("Alibaba video task failed (task=%s): %s", task_id, error)
            return fail(f"Alibaba Token Plan video generation failed: {error}", "task_failed")
        assert output is not None

        remote = str(output.get("video_url") or "").strip()
        if not remote:
            return fail("Task succeeded but carried no video_url", "empty_response")

        video_ref = remote
        if tpm.option("video_gen", "download", True):
            try:
                video_ref = str(save_url_video(remote, prefix=f"alibaba_{family_id}"))
            except Exception as exc:
                logger.warning("Could not cache the generated video (%s); returning the remote URL", exc)

        usage = output.get("usage") if isinstance(output.get("usage"), dict) else {}
        effective_duration = int(clamped) if clamped and clamped > 0 else int(usage.get("output_video_duration") or 5)
        return success_response(
            video=video_ref,
            model=family_id,
            prompt=prompt,
            modality=modality,
            aspect_ratio=parameters.get("ratio", ""),
            duration=effective_duration,
            provider=self.name,
            extra=tpm.merge_extra(
                {"task_id": task_id, "family": family_id,
                 "resolution": parameters.get("resolution"),
                 "source_url": None if video_ref == remote else remote},
                {"fps": usage.get("fps"), "sr": usage.get("SR"), "ratio_actual": usage.get("ratio"),
                 "duration_actual": usage.get("output_video_duration")} if usage else None,
            ),
        )


def register(ctx: Any) -> None:
    """Plugin entry point — wire the provider into the video-gen registry."""
    ctx.register_video_gen_provider(AlibabaTokenPlanVideoGenProvider())
