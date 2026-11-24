#!/usr/bin/env python3
"""Paid smoke test for the Alibaba Token Plan providers — spends plan credits.

Run it with an interpreter that can import Hermes core (the python Hermes itself runs on):

    python scripts/smoke.py --video      # 3 s 480P clip, ~2 min
    python scripts/smoke.py --image      # one 1K image, ~30 s
    python scripts/smoke.py --both

The key is read like the plugins read it: ``ALIBABA_TOKEN_PLAN_MEDIA_API_KEY`` from the
environment, else from ``$HERMES_HOME/.env``, else from ``~/.alibaba/token-plan-media.key``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEY_VAR = "ALIBABA_TOKEN_PLAN_MEDIA_API_KEY"


def _load_env_file() -> None:
    """Fill KEY_VAR from $HERMES_HOME/.env when the process environment has no key."""
    if os.environ.get(KEY_VAR):
        return
    home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    env_file = home / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith(f"{KEY_VAR}="):
            os.environ[KEY_VAR] = line.split("=", 1)[1].strip().strip("'\"")
            return


def _report(label: str, result: dict) -> bool:
    ok = bool(result.get("success"))
    print(f"\n[{label}] {'OK' if ok else 'FAILED'} in the API call")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return ok


def run_video() -> bool:
    from agent.video_gen_registry import get_active_provider

    provider = get_active_provider()
    print(f"video provider: {provider.name} | available: {provider.is_available()}")
    started = time.time()
    result = provider.generate(
        "A paper boat drifting down a rain puddle in a city alley, cinematic, soft morning light",
        duration=3, resolution="480p", aspect_ratio="16:9",
    )
    print(f"elapsed: {time.time() - started:.0f}s")
    return _report("video", result)


def run_image() -> bool:
    from agent.image_gen_registry import get_active_provider

    provider = get_active_provider()
    print(f"image provider: {provider.name} | available: {provider.is_available()}")
    started = time.time()
    result = provider.generate("A paper boat on a rain puddle, cinematic morning light", "landscape")
    print(f"elapsed: {time.time() - started:.0f}s")
    return _report("image", result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", action="store_true", help="generate one image")
    parser.add_argument("--video", action="store_true", help="generate one 3 s 480P clip")
    parser.add_argument("--both", action="store_true", help="both of the above")
    args = parser.parse_args()
    if not (args.image or args.video or args.both):
        parser.error("pick --image, --video or --both (these calls cost credits)")

    _load_env_file()
    from hermes_cli.plugins import get_plugin_manager

    get_plugin_manager().discover_and_load(force=True)

    results = []
    if args.image or args.both:
        results.append(run_image())
    if args.video or args.both:
        results.append(run_video())
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
