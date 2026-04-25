from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI


load_dotenv()


def _normalize_azure_endpoint(endpoint: str) -> str:
    value = endpoint.strip()
    if not value:
        return value

    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return value.rstrip("/")

    # Accept full Azure OpenAI URLs and reduce them to resource root.
    # Example input:
    # https://my-resource.cognitiveservices.azure.com/openai/deployments/x/chat/completions?api-version=...
    # Output:
    # https://my-resource.cognitiveservices.azure.com
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def get_openai_client() -> Any:
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if azure_api_key and azure_endpoint:
        return AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=_normalize_azure_endpoint(azure_endpoint),
            api_version=azure_api_version,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set OPENAI_API_KEY for OpenAI or "
            "AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT for Azure OpenAI."
        )
    return OpenAI(api_key=api_key)
