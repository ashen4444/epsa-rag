from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_name: str | None = None


class ChatLLM(Protocol):
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...


class OpenAIChatClient:
    def __init__(
        self,
        *,
        model_name: str = "gpt-4o-mini",
        timeout: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for OpenAIChatClient. "
                "Install it with: pip install openai"
            ) from exc

        self.model_name = model_name
        self._client = OpenAI(api_key=api_key, timeout=timeout)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**payload)

        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        return LLMResponse(
            content=content.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model_name=getattr(response, "model", self.model_name),
        )