import pytest

import hermes_tpm as tpm


def test_media_url_passes_remote_sources_through():
    assert tpm.media_url("https://cdn.example.com/a.png") == "https://cdn.example.com/a.png"
    assert tpm.media_url("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"


def test_media_url_inlines_a_local_file(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    url = tpm.media_url(str(image))

    assert url.startswith("data:image/png;base64,")
    assert url.endswith("iVBORw0KGgo=")


def test_media_url_rejects_unknown_extensions(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("nope")

    with pytest.raises(ValueError, match="unsupported image extension"):
        tpm.media_url(str(note))


def test_media_url_rejects_missing_files(tmp_path):
    with pytest.raises(ValueError, match="neither an http"):
        tpm.media_url(str(tmp_path / "missing.png"))

    with pytest.raises(ValueError, match="empty"):
        tpm.media_url("   ")


def test_api_key_is_independent_of_other_provider_secrets(monkeypatch, tmp_path):
    monkeypatch.delenv(tpm.KEY_ENV, raising=False)
    monkeypatch.setenv(tpm.KEY_FILE_ENV, str(tmp_path / "absent.key"))
    monkeypatch.setenv("HERMES_CUSTOM_TOKEN_PLAN_AP_SOUTHEAST_1_MAAS_ALIYUNCS_COM_API_KEY", "sk-chat-provider")

    assert tpm.api_key() == ""
    assert tpm.key_source() == ""
    assert tpm.KEY_ENV in tpm.missing_key_message()


def test_api_key_prefers_env_then_key_file(monkeypatch, tmp_path):
    key_file = tmp_path / "token-plan-media.key"
    key_file.write_text("# Alibaba Token Plan media key\nsk-sp-from-file\n")
    key_file.chmod(0o600)
    monkeypatch.setenv(tpm.KEY_FILE_ENV, str(key_file))
    monkeypatch.delenv(tpm.KEY_ENV, raising=False)

    assert tpm.api_key() == "sk-sp-from-file"
    assert tpm.key_source() == f"file:{key_file}"

    monkeypatch.setenv(tpm.KEY_ENV, "sk-sp-from-env")
    assert tpm.api_key() == "sk-sp-from-env"
    assert tpm.key_source() == f"env:{tpm.KEY_ENV}"


def test_base_url_config_and_env(monkeypatch):
    monkeypatch.setattr(tpm, "raw_section", lambda category: {})
    monkeypatch.delenv(tpm.BASE_URL_ENV, raising=False)
    assert tpm.base_url("image_gen") == tpm.DEFAULT_BASE_URL

    monkeypatch.setenv(tpm.BASE_URL_ENV, "https://example.maas.aliyuncs.com/")
    assert tpm.base_url("image_gen") == "https://example.maas.aliyuncs.com"

    monkeypatch.setattr(
        tpm, "raw_section", lambda category: {tpm.PROVIDER_NAME: {"base_url": "https://ws-1.example/"}}
    )
    assert tpm.base_url("video_gen") == "https://ws-1.example"


def test_option_reads_provider_section(monkeypatch):
    monkeypatch.setattr(tpm, "raw_section", lambda category: {tpm.PROVIDER_NAME: {"watermark": True, "poll_interval": "3"}})

    assert tpm.option("video_gen", "watermark", False) is True
    assert tpm.option("video_gen", "poll_interval", 8.0) == 3.0
    assert tpm.option("video_gen", "download", True) is True


def test_merge_extra_drops_none_values():
    assert tpm.merge_extra({"a": 1, "b": None}, {"c": 3}, None) == {"a": 1, "c": 3}
