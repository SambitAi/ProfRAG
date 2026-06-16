from __future__ import annotations

from pydantic import BaseModel, Field

from core.user_store import MIN_PASSWORD_LENGTH


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH)
    security_question: str = Field(min_length=1)
    security_answer: str = Field(min_length=1)


class RegisterResponse(BaseModel):
    email: str
    user_key: str


class LoginRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class MeResponse(BaseModel):
    email: str
    user_key: str


class ResetStartRequest(BaseModel):
    email: str = Field(min_length=3)


class ResetStartResponse(BaseModel):
    security_question: str


class ResetFinishRequest(BaseModel):
    email: str = Field(min_length=3)
    security_answer: str = Field(min_length=1)
    new_password: str = Field(min_length=MIN_PASSWORD_LENGTH)
