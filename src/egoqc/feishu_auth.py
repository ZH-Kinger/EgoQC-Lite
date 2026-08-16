from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


AUTHORIZE_URL = "https://accounts.feishu.cn/open-apis/authen/v1/authorize"
TOKEN_URL = "https://open.feishu.cn/open-apis/authen/v2/oauth/token"
USER_INFO_URL = "https://open.feishu.cn/open-apis/authen/v1/user_info"


class FeishuAuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeishuAuthConfig:
    app_id: str
    app_secret: str
    redirect_uri: str
    allowed_tenant_keys: tuple[str, ...] = ()
    session_ttl_seconds: int = 43200

    @classmethod
    def from_env(cls) -> Optional["FeishuAuthConfig"]:
        app_id = os.environ.get("FEISHU_APP_ID", "").strip()
        app_secret = os.environ.get("FEISHU_APP_SECRET", "").strip()
        redirect_uri = os.environ.get("FEISHU_REDIRECT_URI", "").strip()
        configured = [bool(app_id), bool(app_secret), bool(redirect_uri)]
        if not any(configured):
            return None
        if not all(configured):
            raise ValueError("FEISHU_APP_ID、FEISHU_APP_SECRET、FEISHU_REDIRECT_URI 必须同时配置")
        tenants = tuple(
            value.strip()
            for value in os.environ.get("FEISHU_ALLOWED_TENANT_KEYS", "").split(",")
            if value.strip()
        )
        return cls(
            app_id=app_id,
            app_secret=app_secret,
            redirect_uri=redirect_uri,
            allowed_tenant_keys=tenants,
            session_ttl_seconds=int(os.environ.get("EGOQC_SESSION_TTL_SECONDS", "43200")),
        )

    @property
    def secure_cookie(self) -> bool:
        return self.redirect_uri.lower().startswith("https://")


def pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def authorization_url(config: FeishuAuthConfig, state: str, verifier: str) -> str:
    return AUTHORIZE_URL + "?" + urlencode({
        "client_id": config.app_id,
        "response_type": "code",
        "redirect_uri": config.redirect_uri,
        "state": state,
        "code_challenge": pkce_challenge(verifier),
        "code_challenge_method": "S256",
    })


def _json_request(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    bearer: Optional[str] = None,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise FeishuAuthError("飞书认证服务请求失败") from exc
    if not isinstance(result, dict):
        raise FeishuAuthError("飞书认证服务返回格式异常")
    if result.get("code", 0) != 0:
        raise FeishuAuthError(str(result.get("error_description") or result.get("msg") or "飞书认证失败"))
    return result


def exchange_code(
    config: FeishuAuthConfig,
    code: str,
    verifier: str,
) -> Dict[str, Any]:
    result = _json_request(
        TOKEN_URL,
        "POST",
        {
            "grant_type": "authorization_code",
            "client_id": config.app_id,
            "client_secret": config.app_secret,
            "code": code,
            "redirect_uri": config.redirect_uri,
            "code_verifier": verifier,
        },
    )
    if not result.get("access_token"):
        raise FeishuAuthError("飞书未返回 user_access_token")
    return result


def fetch_user_info(config: FeishuAuthConfig, access_token: str) -> Dict[str, Any]:
    result = _json_request(USER_INFO_URL, bearer=access_token)
    profile = result.get("data")
    if not isinstance(profile, dict) or not profile.get("open_id"):
        raise FeishuAuthError("飞书用户信息缺少 open_id")
    tenant_key = str(profile.get("tenant_key") or "")
    if config.allowed_tenant_keys and tenant_key not in config.allowed_tenant_keys:
        raise FeishuAuthError("当前飞书租户不在允许列表")
    return profile
