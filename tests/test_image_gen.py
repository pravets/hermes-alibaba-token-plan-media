import pytest

import hermes_tpm as tpm
import image_gen


@pytest.fixture(autouse=True)
def no_config(monkeypatch):
    monkeypatch.setattr(tpm, "raw_section", lambda category: {})
    monkeypatch.setenv(tpm.KEY_ENV, "sk-sp-test")
    monkeypatch.delenv("ALIBABA_IMAGE_MODEL", raising=False)


@pytest.fixture
def provider():
    return image_gen.AlibabaTokenPlanImageGenProvider()


def _api_payload(**overrides):
    payload = {
        "output": {"choices": [{"finish_reason": "stop", "message": {"content": [{"image": "https://oss/x.png"}]}}]},
        "usage": {"image_count": 1, "width": 2048, "height": 1152},
        "request_id": "req-1",
    }
    payload.update(overrides)
    return payload


def _capture(monkeypatch, response, *, error=None):
    calls = []

    def fake(method, url, key, *, payload=None, headers=None, timeout=120.0):
        calls.append({"method": method, "url": url, "key": key, "payload": payload,
                      "headers": headers, "timeout": timeout})
        return (response, error) if error is None else (None, error)

    monkeypatch.setattr(tpm, "request_json", fake)
    return calls


def test_catalog_and_setup_schema(provider):
    ids = [row["id"] for row in provider.list_models()]
    assert ids == ["qwen-image-3.0-pro", "wan2.7-image-pro", "wan2.7-image"]
    assert provider.default_model() == "qwen-image-3.0-pro"
    assert provider.name == "alibaba-token-plan"

    schema = provider.get_setup_schema()
    assert schema["env_vars"][0]["key"] == "ALIBABA_TOKEN_PLAN_MEDIA_API_KEY"
    assert schema["badge"] == "subscription"


def test_capabilities_declare_editing(provider):
    caps = provider.capabilities()
    assert caps["modalities"] == ["text", "image"]
    assert caps["max_source_images"] == 9
    assert caps["max_reference_images"] == 8


def test_text_to_image_landscape(monkeypatch, provider, tmp_path):
    saved = tmp_path / "out.png"
    cached = []
    monkeypatch.setattr(image_gen, "save_url_image", lambda url, prefix: cached.append((url, prefix)) or saved)
    calls = _capture(monkeypatch, _api_payload())

    result = provider.generate("a corgi in a spacesuit")

    assert result["success"] is True
    assert result["image"] == str(saved)
    assert result["modality"] == "text"
    assert result["aspect_ratio"] == "landscape"
    assert result["width"] == 2048 and result["height"] == 1152
    assert result["source_url"] == "https://oss/x.png"
    assert cached == [("https://oss/x.png", "alibaba_qwen-image-3.0-pro")]

    call = calls[0]
    assert call["method"] == "POST"
    assert call["url"] == f"{tpm.DEFAULT_BASE_URL}{tpm.IMAGE_PATH}"
    assert call["key"] == "sk-sp-test"
    assert call["payload"]["model"] == "qwen-image-3.0-pro"
    assert call["payload"]["input"]["messages"][0]["role"] == "user"
    assert call["payload"]["input"]["messages"][0]["content"] == [{"text": "a corgi in a spacesuit"}]
    assert call["payload"]["parameters"] == {"size": "2048*1152", "n": 1, "watermark": False, "prompt_extend": True}


def test_aspect_ratio_drives_pixel_size(monkeypatch, provider):
    monkeypatch.setattr(image_gen, "save_url_image", lambda url, prefix: "/cache/x.png")
    calls = _capture(monkeypatch, _api_payload())

    provider.generate("a portrait", "portrait")
    provider.generate("a square", "square")
    provider.generate("a weird aspect", "sideways")

    sizes = [call["payload"]["parameters"]["size"] for call in calls]
    assert sizes == ["1152*2048", "2048*2048", "2048*1152"]


def test_edit_request_puts_source_images_in_content(monkeypatch, provider, tmp_path):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(image_gen, "save_url_image", lambda url, prefix: "/cache/x.png")
    calls = _capture(monkeypatch, _api_payload())

    result = provider.generate("make it night", image_url="https://cdn/base.png", reference_image_urls=[str(ref)])

    content = calls[0]["payload"]["input"]["messages"][0]["content"]
    assert result["modality"] == "image"
    assert content[0] == {"text": "make it night"}
    assert content[1] == {"image": "https://cdn/base.png"}
    assert content[2]["image"].startswith("data:image/png;base64,")


def test_too_many_source_images(provider):
    result = provider.generate("x", reference_image_urls=[f"https://cdn/{i}.png" for i in range(10)])

    assert result["success"] is False
    assert result["error_type"] == "too_many_references"


def test_configured_size_and_model_win(monkeypatch, provider):
    monkeypatch.setattr(tpm, "raw_section", lambda category: {
        tpm.PROVIDER_NAME: {"model": "wan2.7-image-pro", "size": "4K", "watermark": True}
    })
    monkeypatch.setattr(image_gen, "save_url_image", lambda url, prefix: "/cache/x.png")
    calls = _capture(monkeypatch, _api_payload())

    provider.generate("a poster", "square")

    assert calls[0]["payload"]["model"] == "wan2.7-image-pro"
    assert calls[0]["payload"]["parameters"]["size"] == "4K"
    assert calls[0]["payload"]["parameters"]["watermark"] is True


def test_api_error_becomes_clean_error_response(monkeypatch, provider):
    monkeypatch.setattr(image_gen, "save_url_image", lambda url, prefix: "/cache/x.png")
    _capture(monkeypatch, None, error="HTTP 400: InvalidParameter: size out of range")

    result = provider.generate("x")

    assert result["success"] is False
    assert result["error_type"] == "api_error"
    assert "InvalidParameter" in result["error"]
    assert result["provider"] == "alibaba-token-plan"


def test_missing_credentials(monkeypatch, provider):
    monkeypatch.delenv(tpm.KEY_ENV, raising=False)
    monkeypatch.setenv(tpm.KEY_FILE_ENV, "/nonexistent/key")

    result = provider.generate("x")

    assert result["success"] is False
    assert result["error_type"] == "missing_credentials"
    assert "ALIBABA_TOKEN_PLAN_MEDIA_API_KEY" in result["error"]


def test_empty_prompt(provider):
    result = provider.generate("   ")

    assert result["success"] is False
    assert result["error_type"] == "invalid_input"


def test_response_without_image(monkeypatch, provider):
    monkeypatch.setattr(image_gen, "save_url_image", lambda url, prefix: "/cache/x.png")
    _capture(monkeypatch, {"output": {"choices": [{"message": {"content": [{"text": "nope"}]}}]}})

    result = provider.generate("x")

    assert result["success"] is False
    assert result["error_type"] == "empty_response"


def test_is_available_follows_key(monkeypatch, provider):
    assert provider.is_available() is True
    monkeypatch.delenv(tpm.KEY_ENV, raising=False)
    monkeypatch.setenv(tpm.KEY_FILE_ENV, "/nonexistent/key")
    assert provider.is_available() is False


def test_register_wires_provider():
    class Ctx:
        def __init__(self):
            self.providers = []

        def register_image_gen_provider(self, provider):
            self.providers.append(provider)

    ctx = Ctx()
    image_gen.register(ctx)

    assert len(ctx.providers) == 1
    assert ctx.providers[0].name == "alibaba-token-plan"
