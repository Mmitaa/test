"""Вызов Claude: один запрос, строгая JSON-схема, стриминг и кэш базы знаний."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import anthropic

from .config import EFFORT_MODELS, FALLBACK_MODELS, Settings
from .jsonstream import JsonStringFieldStreamer
from .kb import KnowledgeBase
from .prompt import RESPONSE_SCHEMA, build_system_prompt, build_user_content

log = logging.getLogger("llm")


class LLMOutputError(RuntimeError):
    """Модель не вернула полный валидный ответ (отказ, обрыв по max_tokens, битый JSON)."""


@dataclass
class Delta:
    text: str


@dataclass
class Final:
    data: dict
    usage: dict
    model: str


class ClaudeEngine:
    def __init__(self, settings: Settings, kb: KnowledgeBase, client: anthropic.AsyncAnthropic | None = None):
        self.settings = settings
        # Один клиент на весь процесс: keep-alive соединение с API переиспользуется между запросами.
        self.client = client or anthropic.AsyncAnthropic(
            timeout=anthropic.Timeout(settings.request_timeout, connect=5.0),
            max_retries=1,
        )
        cache_control = {"type": "ephemeral"}
        if settings.cache_ttl == "1h":
            cache_control["ttl"] = "1h"
        # База знаний целиком в системном промпте с точкой кэширования: без векторной БД,
        # без эмбеддингов, без лишнего сетевого запроса на поиск.
        self.system = [{"type": "text", "text": build_system_prompt(kb), "cache_control": cache_control}]

    def _params(self, user_content: str, max_tokens: int | None = None) -> dict:
        model = self.settings.model
        output_config: dict = {"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}}
        if model in EFFORT_MODELS:
            output_config["effort"] = self.settings.effort
        params: dict = {
            "model": model,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "system": self.system,
            "messages": [{"role": "user", "content": user_content}],
            "output_config": output_config,
        }
        if model in FALLBACK_MODELS:
            # Если классификатор безопасности отклонит запрос, API сам повторит его на резервной модели.
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"
        return params

    async def stream(self, message: str, deal: dict | None, history: list[dict]) -> AsyncIterator[Delta | Final]:
        params = self._params(build_user_content(message, deal, history))
        reply = JsonStringFieldStreamer("reply")
        async with self.client.beta.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                delta = reply.feed(text)
                if delta:
                    yield Delta(delta)
            final = await stream.get_final_message()

        if final.stop_reason == "refusal":
            raise LLMOutputError("refusal")
        if final.stop_reason == "max_tokens":
            raise LLMOutputError("max_tokens")
        raw = "".join(b.text for b in final.content if b.type == "text")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"invalid json: {exc}") from exc
        yield Final(data=data, usage=_usage(final.usage), model=final.model)

    async def prewarm(self) -> dict | None:
        """Прогрев: заранее записывает системный промпт в кэш API и открывает соединение.

        Запрос повторяет форму боевого (та же схема и effort), иначе кэш-запись не совпадёт
        с префиксом реальных запросов; ответ обрезается на 1 токене и выбрасывается.
        """
        try:
            params = self._params(build_user_content("warmup", None, []), max_tokens=1)
            msg = await self.client.beta.messages.create(**params)
            usage = _usage(msg.usage)
            log.info("prompt cache prewarmed: %s", usage)
            return usage
        except anthropic.APIError as exc:
            log.warning("prewarm failed: %s", exc)
            return None


def _usage(u) -> dict:
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }
