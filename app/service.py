"""Оркестрация: кэш ответов → Claude (стриминг) → резервный движок, плюс метрики."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator

import anthropic

from .config import Settings, estimate_cost
from .kb import KnowledgeBase, normalize
from .llm import ClaudeEngine, Delta, LLMOutputError
from .offline import answer_offline

log = logging.getLogger("service")


class AnswerCache:
    """LRU-кэш с TTL для повторяющихся типовых вопросов: ответ за ~0 мс и $0."""

    def __init__(self, size: int, ttl: float):
        self.size, self.ttl = size, ttl
        self._data: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    def get(self, key: str) -> dict | None:
        item = self._data.get(key)
        if not item:
            return None
        ts, value = item
        if time.monotonic() - ts > self.ttl:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def put(self, key: str, value: dict) -> None:
        self._data[key] = (time.monotonic(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.size:
            self._data.popitem(last=False)


def cache_key(message: str, deal: dict | None) -> str:
    deal = deal or {}
    tags = ",".join(sorted(t.lower() for t in deal.get("tags", [])))
    return f"{normalize(message)}|{deal.get('client_name', '')}|{tags}"


def enforce_upsell_rules(kb: KnowledgeBase, data: dict, deal: dict | None) -> dict:
    """Детерминированная страховка поверх модели: только товары из базы и не то, что у клиента уже есть."""
    up = data.get("upsell") or {}
    tags = {t.lower() for t in (deal or {}).get("tags", [])}
    rule = kb.rule(up.get("rule_id", ""))
    if up.get("rule_id") not in (None, "", "none") and (rule is None or (rule.exclude and rule.exclude in tags)):
        data["upsell"] = {
            "rule_id": "none",
            "product": "",
            "reason": "Подсказка отфильтрована: товара нет в правилах или он уже есть у клиента.",
            "script": "",
            "priority": "none",
        }
    elif rule is not None:
        up["product"] = up.get("product") or rule.product
    return data


class SupportService:
    def __init__(self, settings: Settings, kb: KnowledgeBase, engine: ClaudeEngine | None = None):
        self.settings = settings
        self.kb = kb
        if engine is None and settings.llm_enabled:
            engine = ClaudeEngine(settings, kb)
        self.engine = engine
        self.cache = AnswerCache(settings.answer_cache_size, settings.answer_cache_ttl)
        self.last_llm_call = 0.0
        self.stats = {"requests": 0, "llm": 0, "cache": 0, "offline": 0, "errors": 0}

    @property
    def mode(self) -> str:
        return "claude" if self.engine else "offline"

    async def stream(self, message: str, deal: dict | None = None, history: list[dict] | None = None) -> AsyncIterator[dict]:
        """События: meta → delta* → result. При сбое ИИ после начала стрима — reset и ответ по базе."""
        history = history or []
        t0 = time.perf_counter()
        self.stats["requests"] += 1

        def ms() -> float:
            return round((time.perf_counter() - t0) * 1000, 1)

        # Кэшируем только первое сообщение диалога: там ответ зависит лишь от вопроса и сделки.
        key = cache_key(message, deal) if not history else None
        if key and (hit := self.cache.get(key)):
            self.stats["cache"] += 1
            yield {"type": "meta", "source": "cache", "model": hit["_model"]}
            yield {"type": "delta", "text": hit["data"]["reply"]}
            elapsed = ms()
            yield {"type": "result", "data": hit["data"], "metrics": {
                "source": "cache", "model": hit["_model"], "ttft_ms": elapsed, "total_ms": elapsed,
                "usage": {}, "cost_usd": 0.0,
            }}
            return

        fallback_reason = None
        if self.engine:
            model = self.settings.model
            yield {"type": "meta", "source": "llm", "model": model}
            ttft = None
            try:
                self.last_llm_call = time.monotonic()
                async for ev in self.engine.stream(message, deal, history):
                    if isinstance(ev, Delta):
                        if ttft is None:
                            ttft = ms()
                        yield {"type": "delta", "text": ev.text}
                        continue
                    data = enforce_upsell_rules(self.kb, ev.data, deal)
                    metrics = {
                        "source": "llm", "model": ev.model, "ttft_ms": ttft if ttft is not None else ms(),
                        "total_ms": ms(), "usage": ev.usage,
                        "cost_usd": estimate_cost(model, ev.usage, self.settings.cache_ttl),
                    }
                    self.stats["llm"] += 1
                    if key and data.get("found_in_kb") and not data.get("escalate"):
                        self.cache.put(key, {"data": data, "_model": ev.model})
                    yield {"type": "result", "data": data, "metrics": metrics}
                    return
            except (anthropic.APIError, LLMOutputError) as exc:
                self.stats["errors"] += 1
                fallback_reason = f"{type(exc).__name__}: {exc}"[:300]
                log.warning("LLM failed, falling back to offline engine: %s", fallback_reason)
                if ttft is not None:
                    yield {"type": "reset"}

        self.stats["offline"] += 1
        data = answer_offline(self.kb, message, deal, history)
        yield {"type": "meta", "source": "offline", "model": "rules"}
        yield {"type": "delta", "text": data["reply"]}
        elapsed = ms()
        metrics = {"source": "offline", "model": "rules", "ttft_ms": elapsed, "total_ms": elapsed,
                   "usage": {}, "cost_usd": 0.0}
        if fallback_reason:
            metrics["fallback_reason"] = fallback_reason
        yield {"type": "result", "data": data, "metrics": metrics}

    async def answer(self, message: str, deal: dict | None = None, history: list[dict] | None = None) -> dict:
        """Нестриминговый вариант (для вебхуков amoCRM и бенчмарка)."""
        result: dict = {}
        async for ev in self.stream(message, deal, history):
            if ev["type"] == "result":
                result = {"data": ev["data"], "metrics": ev["metrics"]}
        return result

    async def keep_warm(self) -> None:
        """Держит кэш промпта тёплым при простое: перезаписывает его чуть раньше, чем истечёт TTL."""
        if not self.engine or not self.settings.prewarm:
            return
        interval = 55 * 60 if self.settings.cache_ttl == "1h" else 4 * 60 + 30
        await self.engine.prewarm()
        self.last_llm_call = time.monotonic()
        while True:
            await asyncio.sleep(30)
            if time.monotonic() - self.last_llm_call >= interval:
                await self.engine.prewarm()
                self.last_llm_call = time.monotonic()
