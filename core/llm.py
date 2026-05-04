from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI


load_dotenv()


_GOOGLE_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class _GoogleADCOpenAIClient:
    """OpenAI-compatible client for Vertex AI using Google ADC access tokens."""

    def __init__(self, project_id: str, location: str, endpoint_id: str) -> None:
        self.project_id = project_id
        self.location = location
        self.endpoint_id = endpoint_id
        self._credentials = None
        self._token_expires_at: datetime | None = None

    def _refresh_token_if_needed(self) -> str:
        now = datetime.now(timezone.utc)
        if self._credentials is None:
            try:
                from google.auth import default
                from google.auth.transport.requests import Request
            except Exception as exc:
                raise RuntimeError(
                    "Google ADC support requires `google-auth`. "
                    "Install it with: pip install google-auth"
                ) from exc

            credentials, _ = default(scopes=[_GOOGLE_SCOPE])
            credentials.refresh(Request())
            self._credentials = credentials
            self._token_expires_at = credentials.expiry
            return credentials.token

        # Refresh if token is missing, expired, or within 2 minutes of expiry.
        needs_refresh = (
            not getattr(self._credentials, "token", None)
            or self._token_expires_at is None
            or now >= (self._token_expires_at - timedelta(minutes=2))
        )
        if needs_refresh:
            from google.auth.transport.requests import Request

            self._credentials.refresh(Request())
            self._token_expires_at = self._credentials.expiry

        return self._credentials.token

    def _build_client(self) -> OpenAI:
        token = self._refresh_token_if_needed()
        base_url = (
            f"https://aiplatform.googleapis.com/v1/projects/{self.project_id}"
            f"/locations/{self.location}/endpoints/{self.endpoint_id}"
        )
        return OpenAI(api_key=token, base_url=base_url)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._build_client(), item)


class _GoogleNativeClient:
    """Google-native client shim that exposes OpenAI-like methods used by this codebase."""

    class _EmbeddingData:
        def __init__(self, embedding: list[float]) -> None:
            self.embedding = embedding

    class _EmbeddingResponse:
        def __init__(self, embedding: list[float]) -> None:
            self.data = [_GoogleNativeClient._EmbeddingData(embedding)]

    class _Message:
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = _GoogleNativeClient._Message(content)

    class _ChatResponse:
        def __init__(self, content: str) -> None:
            self.choices = [_GoogleNativeClient._Choice(content)]

    class _Embeddings:
        def __init__(self, outer: "_GoogleNativeClient") -> None:
            self._outer = outer

        def create(self, *, model: str, input: list[str]) -> "_GoogleNativeClient._EmbeddingResponse":
            if not input:
                raise ValueError("input cannot be empty")
            text = input[0]
            response = self._outer._client.models.embed_content(model=model, contents=text)
            embeddings = getattr(response, "embeddings", None) or []
            if not embeddings:
                raise RuntimeError("Google embed_content returned no embeddings.")
            values = getattr(embeddings[0], "values", None)
            if values is None:
                values = embeddings[0].get("values", []) if isinstance(embeddings[0], dict) else []
            return _GoogleNativeClient._EmbeddingResponse(list(values))

    class _ChatCompletions:
        def __init__(self, outer: "_GoogleNativeClient") -> None:
            self._outer = outer

        def create(
            self,
            *,
            model: str,
            messages: list[dict[str, str]],
            temperature: float = 0,
        ) -> "_GoogleNativeClient._ChatResponse":
            prompt_parts: list[str] = []
            for msg in messages:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                if content:
                    prompt_parts.append(f"{role}: {content}")
            prompt = "\n\n".join(prompt_parts).strip()
            if not prompt:
                prompt = "USER: "
            response = self._outer._client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": temperature},
            )
            text = getattr(response, "text", None) or ""
            if not text and hasattr(response, "candidates"):
                try:
                    parts = response.candidates[0].content.parts
                    text = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None))
                except Exception:
                    text = ""
            return _GoogleNativeClient._ChatResponse(text)

    class _Chat:
        def __init__(self, outer: "_GoogleNativeClient") -> None:
            self.completions = _GoogleNativeClient._ChatCompletions(outer)

    def __init__(self, *, api_key: str | None, project_id: str | None, location: str) -> None:
        try:
            from google import genai
        except Exception as exc:
            raise RuntimeError(
                "Google native provider requires `google-genai`. "
                "Install it with: pip install google-genai"
            ) from exc

        if api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            if not project_id:
                try:
                    from google.auth import default

                    _, detected_project = default(scopes=[_GOOGLE_SCOPE])
                    project_id = detected_project
                except Exception:
                    project_id = None
            if not project_id:
                raise RuntimeError(
                    "Google native provider requires GOOGLE_CLOUD_PROJECT (or detectable ADC project)."
                )
            self._client = genai.Client(vertexai=True, project=project_id, location=location)

        self.embeddings = _GoogleNativeClient._Embeddings(self)
        self.chat = _GoogleNativeClient._Chat(self)


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


def get_openai_client(purpose: str = "default") -> Any:
    provider = ""
    if purpose == "summarizer":
        provider = (os.getenv("SUMMARIZER_LLM_PROVIDER") or "").strip().lower()
    if not provider:
        provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    google_project_id = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GOOGLE_PROJECT_ID")
        or os.getenv("GCP_PROJECT")
    )
    google_location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    google_endpoint_id = os.getenv("GOOGLE_OPENAI_ENDPOINT_ID", "openapi")

    if provider == "azure":
        if not (azure_api_key and azure_endpoint):
            raise RuntimeError("LLM_PROVIDER=azure requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.")
        return AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=_normalize_azure_endpoint(azure_endpoint),
            api_version=azure_api_version,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if provider == "openai":
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY.")
        return OpenAI(api_key=api_key)

    if provider == "google_native":
        return _GoogleNativeClient(
            api_key=google_api_key,
            project_id=google_project_id,
            location=google_location,
        )

    if provider == "google":
        if google_api_key:
            return OpenAI(
                api_key=google_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        if not google_project_id:
            try:
                from google.auth import default
                _, detected_project = default(scopes=[_GOOGLE_SCOPE])
                google_project_id = detected_project
            except Exception:
                google_project_id = None
        if not google_project_id:
            raise RuntimeError(
                "LLM_PROVIDER=google requires GOOGLE_API_KEY or Google ADC "
                "(GOOGLE_CLOUD_PROJECT + gcloud auth application-default login)."
            )
        return _GoogleADCOpenAIClient(
            project_id=google_project_id,
            location=google_location,
            endpoint_id=google_endpoint_id,
        )

    if azure_api_key and azure_endpoint:
        return AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=_normalize_azure_endpoint(azure_endpoint),
            api_version=azure_api_version,
        )

    if api_key:
        return OpenAI(api_key=api_key)

    if google_api_key:
        # Uses Google's OpenAI-compatible endpoint.
        # Model names are read from app config (chat.model / embeddings.model).
        return OpenAI(
            api_key=google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    if (os.getenv("LLM_PROVIDER") or "").strip().lower() == "google_native":
        return _GoogleNativeClient(
            api_key=None,
            project_id=google_project_id,
            location=google_location,
        )

    # Google ADC path (no API key): Vertex AI OpenAI-compatible endpoint.
    if not google_project_id:
        # Try to detect the project from ADC metadata.
        try:
            from google.auth import default

            _, detected_project = default(scopes=[_GOOGLE_SCOPE])
            google_project_id = detected_project
        except Exception:
            google_project_id = None

    if google_project_id:
        return _GoogleADCOpenAIClient(
            project_id=google_project_id,
            location=google_location,
            endpoint_id=google_endpoint_id,
        )

    raise RuntimeError(
        "No API key found. Set one of: OPENAI_API_KEY, "
        "AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT, "
        "or Google ADC (GOOGLE_CLOUD_PROJECT + gcloud ADC login), "
        "or GOOGLE_API_KEY."
    )
