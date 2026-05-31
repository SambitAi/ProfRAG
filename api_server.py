from __future__ import annotations

import uvicorn

from api.app import app
from api.settings import load_api_settings


if __name__ == "__main__":
    settings = load_api_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)

