from urllib.parse import parse_qs, urlparse

import pytest

from egoqc.feishu_auth import FeishuAuthConfig, authorization_url, pkce_challenge


def test_feishu_authorization_url_uses_current_endpoint_and_pkce():
    config = FeishuAuthConfig("cli_test", "secret", "http://127.0.0.1:8767/auth/callback")
    verifier = "A" * 64
    url = urlparse(authorization_url(config, "state-1", verifier))
    query = parse_qs(url.query)
    assert url.netloc == "accounts.feishu.cn"
    assert url.path == "/open-apis/authen/v1/authorize"
    assert query["client_id"] == ["cli_test"]
    assert query["state"] == ["state-1"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == [pkce_challenge(verifier)]


def test_feishu_config_requires_complete_credentials(monkeypatch):
    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)
    monkeypatch.delenv("FEISHU_REDIRECT_URI", raising=False)
    with pytest.raises(ValueError, match="必须同时配置"):
        FeishuAuthConfig.from_env()


def test_feishu_tenant_allowlist_from_environment(monkeypatch):
    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret")
    monkeypatch.setenv("FEISHU_REDIRECT_URI", "https://review.example.com/auth/callback")
    monkeypatch.setenv("FEISHU_ALLOWED_TENANT_KEYS", "tenant-a, tenant-b")
    config = FeishuAuthConfig.from_env()
    assert config is not None
    assert config.allowed_tenant_keys == ("tenant-a", "tenant-b")
    assert config.secure_cookie
