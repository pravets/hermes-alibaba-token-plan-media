"""Shared helpers for the Alibaba Token Plan media provider plugins.

Both plugins (``image_gen`` / ``video_gen``) talk to the *native* DashScope paths of
the Token Plan host. The plan has no OpenAI-compatible surface for images or video:
``/compatible-mode/v1/images/generations`` and ``.../videos/generations`` return 404,
only text models live there.

Credential rules baked in here:

* The media credential is **independent** of any chat/model-provider key. It is read
  from ``ALIBABA_TOKEN_PLAN_MEDIA_API_KEY`` (prompted by ``hermes tools`` and stored
  in ``$HERMES_HOME/.env``) or, failing that, from a key file
  (``~/.alibaba/token-plan-media.key``, chmod 600 recommended).
* Nothing here ever falls back to ``HERMES_CUSTOM_*`` or other provider secrets.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

PROVIDER_NAME = "alibaba-token-plan"
DISPLAY_NAME = "Alibaba Token Plan"

# Personal Edition lives in Singapore only; a workspace-dedicated host is set via config.
DEFAULT_BASE_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com"
BASE_URL_ENV = "ALIBABA_TOKEN_PLAN_BASE_URL"

KEY_ENV = "ALIBABA_TOKEN_PLAN_MEDIA_API_KEY"
KEY_FILE_ENV = "ALIBABA_TOKEN_PLAN_MEDIA_KEY_FILE"
DEFAULT_KEY_FILE = "~/.alibaba/token-plan-media.key"
CONSOLE_KEYS_URL = "https://modelstudio.console.alibabacloud.com/subscription/token-plan/personal"

IMAGE_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
VIDEO_SUBMIT_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
TASK_PATH = "/api/v1/tasks/{task_id}"

MEDIA_MAX_BYTES = 20 * 1024 * 1024
_MEDIA_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


# ── configuration ──────────────────────────────────────────────────────────────


def raw_section(category: str) -> dict[str, Any]:
    """The whole ``image_gen`` / ``video_gen`` section from config.yaml; ``{}`` on any failure."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:  # config is optional at import time
        logger.debug("Could not load Hermes config: %s", exc)
        return {}
    section = cfg.get(category) if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def config_section(category: str) -> dict[str, Any]:
    """``<category>.<provider>`` from config.yaml (``alibaba`` accepted as an alias)."""
    section = raw_section(category)
    for key in (PROVIDER_NAME, "alibaba", "alibaba_token_plan"):
        own = section.get(key)
        if isinstance(own, dict):
            return own
    return {}


def base_url(category: str) -> str:
    """Provider config → ``ALIBABA_TOKEN_PLAN_BASE_URL`` → Singapore default."""
    raw = config_section(category).get("base_url") or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL
    return str(raw).strip().rstrip("/")


def option(category: str, key: str, default: Any) -> Any:
    """One provider-scoped knob with a sane default (wrong types fall back silently)."""
    value = config_section(category).get(key, default)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return value if isinstance(value, str) and value.strip() else default


def key_file_path() -> Path:
    raw = os.environ.get(KEY_FILE_ENV) or DEFAULT_KEY_FILE
    return Path(os.path.expanduser(raw))


# ── credentials ────────────────────────────────────────────────────────────────


def api_key() -> str:
    """Env var first (profile-aware), then the dedicated key file. Never another vendor's key."""
    value = ""
    try:
        from agent.secret_scope import get_secret_str

        value = (get_secret_str(KEY_ENV) or "").strip()
    except Exception:
        value = (os.environ.get(KEY_ENV) or "").strip()
    if value:
        return value

    path = key_file_path()
    try:
        if not path.is_file():
            return ""
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s key file %s: %s", DISPLAY_NAME, path, exc)
        return ""
    for line in raw.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                if (path.stat().st_mode & 0o077) and os.name == "posix":
                    logger.warning("%s key file %s is group/world-readable; run: chmod 600 %s", DISPLAY_NAME, path, path)
            except OSError:
                pass
            return line
    return ""


def key_source() -> str:
    """Human-readable description of where the credential came from (for pickers/errors)."""
    try:
        from agent.secret_scope import get_secret_str

        if (get_secret_str(KEY_ENV) or "").strip():
            return f"env:{KEY_ENV}"
    except Exception:
        if (os.environ.get(KEY_ENV) or "").strip():
            return f"env:{KEY_ENV}"
    if api_key():
        return f"file:{key_file_path()}"
    return ""


def missing_key_message() -> str:
    return (
        f"{DISPLAY_NAME} media API key is not configured. Set {KEY_ENV} (run `hermes tools` → "
        f"Image/Video Generation → {DISPLAY_NAME}, or put it in $HERMES_HOME/.env) or write the key "
        f"to {key_file_path()}. The key is the Token Plan key starting with `sk-sp-` from "
        f"{CONSOLE_KEYS_URL}."
    )


# ── HTTP ───────────────────────────────────────────────────────────────────────


def _api_error(payload: Any, status: Optional[int]) -> str:
    if isinstance(payload, dict):
        code = payload.get("code") or (payload.get("output") or {}).get("code")
        message = payload.get("message") or (payload.get("output") or {}).get("message")
        if code or message:
            prefix = f"HTTP {status}: " if status else ""
            return f"{prefix}{code or 'error'}: {message or 'no message'}"
    return f"HTTP {status}" if status else "request failed"


def request_json(
    method: str,
    url: str,
    key: str,
    *,
    payload: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 120.0,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """``(json, error)`` — never raises for HTTP/network/parse problems."""
    request_headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    request_headers.update(headers or {})
    try:
        response = requests.request(method, url, json=payload, headers=request_headers, timeout=timeout)
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}"

    try:
        data = response.json()
    except ValueError:
        data = None
    if not response.ok:
        return None, _api_error(data, response.status_code)
    if not isinstance(data, dict):
        return None, f"HTTP {response.status_code}: response was not a JSON object"
    if not data.get("output") and (data.get("code") or data.get("message")):
        return None, _api_error(data, response.status_code)
    return data, None


# ── media inputs ───────────────────────────────────────────────────────────────


def media_url(source: str) -> str:
    """Public http(s)/data URI passes through; a local image becomes a ``data:`` URI.

    The native API accepts either form everywhere a media ``url`` is expected, so local
    files work without uploading them first.
    """
    value = (source or "").strip()
    if not value:
        raise ValueError("media source is empty")
    lowered = value.lower()
    if lowered.startswith(("http://", "https://", "data:")):
        return value

    path = Path(os.path.expanduser(value))
    if not path.is_file():
        raise ValueError(f"media source is neither an http(s) URL nor an existing file: {value}")
    size = path.stat().st_size
    if size > MEDIA_MAX_BYTES:
        raise ValueError(f"media file {path} is {size} bytes; the API accepts at most {MEDIA_MAX_BYTES}")
    mime = _MEDIA_MIMES.get(path.suffix.lower())
    if not mime:
        raise ValueError(f"unsupported image extension {path.suffix!r} (png/jpg/jpeg/webp/bmp)")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def merge_extra(*mappings: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Flatten non-empty extras (``None`` values dropped)."""
    merged: dict[str, Any] = {}
    for mapping in mappings:
        if isinstance(mapping, dict):
            merged.update({k: v for k, v in mapping.items() if v is not None})
    return merged
