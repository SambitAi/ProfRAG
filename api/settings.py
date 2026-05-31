from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class APISettings:
    config_path: str
    host: str
    port: int
    auth_token: str
    rate_limit_per_minute: int
    max_request_bytes: int


def load_api_settings() -> APISettings:
    config_path = os.getenv("PROFRAG_CONFIG_PATH", "config/app_config.yaml")
    host = os.getenv("PROFRAG_API_HOST", "127.0.0.1")
    port = int(os.getenv("PROFRAG_API_PORT", "8001"))
    auth_token = os.getenv("PROFRAG_API_TOKEN", "")
    rate_limit = int(os.getenv("PROFRAG_API_RATE_LIMIT_PER_MINUTE", "120"))
    max_request_bytes = int(os.getenv("PROFRAG_API_MAX_REQUEST_BYTES", str(20 * 1024 * 1024)))
    return APISettings(
        config_path=str(Path(config_path)),
        host=host,
        port=port,
        auth_token=auth_token,
        rate_limit_per_minute=rate_limit,
        max_request_bytes=max_request_bytes,
    )
