import pytest

import hermes_tpm as tpm
import video_gen


@pytest.fixture(autouse=True)
def no_config(monkeypatch):
    monkeypatch.setattr(tpm, "raw_section", lambda category: {})
    monkeypatch.setenv(tpm.KEY_ENV, "sk-sp-test")
    monkeypatch.delenv("ALIBABA_VIDEO_MODEL", raising=False)


@pytest.fixture
def provider():
    return video_gen.AlibabaTokenPlanVideoGenProvider()


def _transport(monkeypatch, *, submit=None, polls=(), submit_error=None):
    calls = []
    pending = list(polls)

    def fake(method, url, key, *, payload=None, headers=None, timeout=120.0):
        calls.append({"method": method, "url": url, "key": key, "payload": payload,
                      "headers": headers, "timeout": timeout})
        if method == "POST":
            if submit_error:
                return None, submit_error
            return submit or {"output": {"task_id": "task-1", "task_status": "PENDING"}}, None
        if pending:
            return pending.pop(0), None
        return {
            "output": {
                "task_id": "task-1",
                "task_status": "SUCCEEDED",
                "video_url": "https://oss.example/x.mp4",
                "usage": {"fps": 30, "SR": 720, "ratio": "16:9", "output_video_duration": 5.0},
            }
        }, None

    monkeypatch.setattr(tpm, "request_json", fake)
    monkeypatch.setattr(video_gen.time, "sleep", lambda seconds: None)
    return calls


def _drain():
    return {"output": {"task_id": "task-1", "task_status": "RUNNING"}}


def test_catalog_and_setup_schema(provider):
    ids = [row["id"] for row in provider.list_models()]
    assert ids == ["wan3.0-video", "wan3.0-video-prime", "happyhorse-1.1-t2v",
                   "happyhorse-1.1-i2v", "happyhorse-1.1-r2v"]
    assert provider.default_model() == "wan3.0-video"
    assert provider.get_setup_schema()["env_vars"][0]["key"] == "ALIBABA_TOKEN_PLAN_MEDIA_API_KEY"


def test_capabilities_are_honest_about_the_api(provider):
    caps = provider.capabilities()
    assert caps["modalities"] == ["text", "image"]
    assert caps["supports_audio"] is True
    assert caps["supports_negative_prompt"] is False
    assert caps["max_duration"] == 30
    assert caps["min_duration"] == 2
    assert "480P" in caps["resolutions"]


def test_text_to_video_submits_async_and_returns_a_local_file(monkeypatch, provider, tmp_path):
    video_path = tmp_path / "out.mp4"
    downloaded = []
    monkeypatch.setattr(video_gen, "save_url_video", lambda url, prefix: downloaded.append((url, prefix)) or video_path)
    calls = _transport(monkeypatch, polls=[_drain()])

    result = provider.generate("a kitten on a rooftop", duration=5, resolution="720p", aspect_ratio="16:9")

    assert result["success"] is True
    assert result["video"] == str(video_path)
    assert result["modality"] == "text"
    assert result["duration"] == 5
    assert result["task_id"] == "task-1"
    assert result["source_url"] == "https://oss.example/x.mp4"
    assert downloaded == [("https://oss.example/x.mp4", "alibaba_wan3.0-video")]

    submit = calls[0]
    assert submit["method"] == "POST"
    assert submit["url"] == f"{tpm.DEFAULT_BASE_URL}{tpm.VIDEO_SUBMIT_PATH}"
    assert submit["headers"] == {"X-DashScope-Async": "enable"}
    assert submit["payload"]["model"] == "wan3.0-video"
    assert submit["payload"]["input"] == {"prompt": "a kitten on a rooftop"}
    assert submit["payload"]["parameters"] == {
        "prompt_extend": True, "watermark": False, "resolution": "720P", "ratio": "16:9", "duration": 5,
    }
    assert calls[1]["method"] == "GET"
    assert calls[1]["url"] == f"{tpm.DEFAULT_BASE_URL}/api/v1/tasks/task-1"


def test_duration_is_clamped_and_zero_means_smart(monkeypatch, provider):
    monkeypatch.setattr(video_gen, "save_url_video", lambda url, prefix: "/cache/x.mp4")
    calls = _transport(monkeypatch)

    provider.generate("long clip", duration=90)
    provider.generate("short clip", duration=1)
    provider.generate("smart clip", duration=0)

    durations = [call["payload"]["parameters"].get("duration") for call in calls if call["method"] == "POST"]
    assert durations == [30, 2, -1]


def test_image_to_video_sends_first_frame(monkeypatch, provider, tmp_path):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(video_gen, "save_url_video", lambda url, prefix: "/cache/x.mp4")
    calls = _transport(monkeypatch, polls=[{"output": {"task_status": "RUNNING"}}])

    result = provider.generate("make it move", image_url=str(frame), duration=5)

    assert result["success"] is True
    assert result["modality"] == "image"
    media = calls[0]["payload"]["input"]["media"]
    assert len(media) == 1
    assert media[0]["type"] == "first_frame"
    assert media[0]["url"].startswith("data:image/png;base64,")


def test_reference_images_use_reference_type(monkeypatch, provider):
    monkeypatch.setattr(video_gen, "save_url_video", lambda url, prefix: "/cache/x.mp4")
    calls = _transport(monkeypatch)

    result = provider.generate("the same character in two scenes", model="happyhorse-1.1-r2v",
                               reference_image_urls=["https://cdn/a.png", "https://cdn/b.png"])

    assert result["success"] is True
    media = calls[0]["payload"]["input"]["media"]
    assert [item["type"] for item in media] == ["reference_image", "reference_image"]
    assert [item["url"] for item in media] == ["https://cdn/a.png", "https://cdn/b.png"]


def test_first_frame_family_drops_the_ratio(monkeypatch, provider):
    monkeypatch.setattr(video_gen, "save_url_video", lambda url, prefix: "/cache/x.mp4")
    calls = _transport(monkeypatch)

    provider.generate("animate", model="happyhorse-1.1-i2v", image_url="https://cdn/a.png",
                      aspect_ratio="9:16", resolution="1080p", duration=15)

    parameters = calls[0]["payload"]["parameters"]
    assert "ratio" not in parameters  # i2v follows the image; the API rejects ratio
    assert parameters["resolution"] == "1080P"
    assert parameters["duration"] == 15
    assert calls[0]["payload"]["model"] == "happyhorse-1.1-i2v"


def test_text_only_family_rejects_an_image(monkeypatch, provider):
    calls = _transport(monkeypatch)

    result = provider.generate("x", model="happyhorse-1.1-t2v", image_url="https://cdn/a.png")

    assert result["success"] is False
    assert result["error_type"] == "modality_unsupported"
    assert calls == []  # nothing was submitted


def test_failed_task_reports_code_and_message(monkeypatch, provider):
    _transport(monkeypatch, polls=[{
        "output": {"task_status": "FAILED", "code": "InvalidParameter", "message": "bad ratio"}
    }])

    result = provider.generate("x")

    assert result["success"] is False
    assert result["error_type"] == "task_failed"
    assert "InvalidParameter" in result["error"] and "bad ratio" in result["error"]


def test_poll_deadline(monkeypatch, provider):
    def fake(method, url, key, *, payload=None, headers=None, timeout=120.0):
        if method == "POST":
            return {"output": {"task_id": "task-1", "task_status": "PENDING"}}, None
        return {"output": {"task_status": "RUNNING"}}, None

    monkeypatch.setattr(tpm, "request_json", fake)
    monkeypatch.setattr(video_gen.time, "sleep", lambda seconds: None)
    clock = iter(range(0, 20_000, 500))
    monkeypatch.setattr(video_gen.time, "monotonic", lambda: next(clock))

    result = provider.generate("x")

    assert result["success"] is False
    assert result["error_type"] == "task_failed"
    assert "did not finish in time" in result["error"]


def test_submit_error_is_reported(monkeypatch, provider):
    _transport(monkeypatch, submit_error="HTTP 401: InvalidApiKey: No API-key provided.")

    result = provider.generate("x")

    assert result["success"] is False
    assert result["error_type"] == "api_error"
    assert "InvalidApiKey" in result["error"]


def test_missing_key(monkeypatch, provider):
    monkeypatch.delenv(tpm.KEY_ENV, raising=False)
    monkeypatch.setenv(tpm.KEY_FILE_ENV, "/nonexistent/key")

    result = provider.generate("x")

    assert result["success"] is False
    assert result["error_type"] == "missing_credentials"


def test_downloading_can_be_disabled(monkeypatch, provider):
    monkeypatch.setattr(tpm, "raw_section", lambda category: {tpm.PROVIDER_NAME: {"download": False}})
    _transport(monkeypatch)

    result = provider.generate("x")

    assert result["video"] == "https://oss.example/x.mp4"
    assert "source_url" not in result


def test_register_wires_provider():
    class Ctx:
        def __init__(self):
            self.providers = []

        def register_video_gen_provider(self, provider):
            self.providers.append(provider)

    ctx = Ctx()
    video_gen.register(ctx)

    assert len(ctx.providers) == 1
    assert ctx.providers[0].name == "alibaba-token-plan"
