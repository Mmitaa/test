"""Фейковый клиент Anthropic: стримит заранее заданный JSON кусочками, как настоящий API."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("AMO_TOKEN", None)

from app.config import Settings  # noqa: E402
from app.kb import load_kb  # noqa: E402

SAMPLE = {
    "reply": "Игорь, здравствуйте! Советуем бленд «Утро» — шоколад и орех, без кислинки.\nПачки 250 г и 1 кг 😊",
    "found_in_kb": True,
    "escalate": False,
    "sentiment": "neutral",
    "kb_refs": ["Кофе в зёрнах"],
    "upsell": {
        "rule_id": "beans_subscription",
        "product": "Подписка «Свежая обжарка»",
        "reason": "Клиент купил машину и регулярно покупает кофе.",
        "script": "Можно оформить подписку со скидкой 15%.",
        "priority": "high",
    },
}


def chunks(text: str, size: int = 7) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


class FakeStream:
    def __init__(self, parts: list[str], stop_reason: str, fail_after: int | None):
        self.parts, self.stop_reason, self.fail_after = parts, stop_reason, fail_after

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def gen():
            for i, p in enumerate(self.parts):
                if self.fail_after is not None and i >= self.fail_after:
                    import anthropic
                    import httpx2

                    raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))
                yield p

        return gen()

    async def get_final_message(self):
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            model="claude-opus-5-5",
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="".join(self.parts))],
            usage=SimpleNamespace(input_tokens=180, output_tokens=150, cache_read_input_tokens=3900,
                                  cache_creation_input_tokens=0),
        )


class FakeMessages:
    def __init__(self, payload: dict, stop_reason: str = "end_turn", fail_after: int | None = None):
        self.payload, self.stop_reason, self.fail_after = payload, stop_reason, fail_after
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(chunks(json.dumps(self.payload, ensure_ascii=False)), self.stop_reason, self.fail_after)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["max_tokens"] == 1:  # прогрев кэша
            return SimpleNamespace(stop_reason="max_tokens", content=[], usage=SimpleNamespace(
                input_tokens=10, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=3900))
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="none\nПочему: ...")],
                               usage=SimpleNamespace(input_tokens=4000, output_tokens=120,
                                                     cache_read_input_tokens=0, cache_creation_input_tokens=0))


class FakeClient:
    def __init__(self, **kwargs):
        self.beta = SimpleNamespace(messages=FakeMessages(**kwargs))


@pytest.fixture
def settings() -> Settings:
    return Settings(model="claude-opus-5-5", prewarm=False)


@pytest.fixture
def kb(settings):
    return load_kb(settings.kb_path)


@pytest.fixture
def cases() -> list[dict]:
    return json.loads((ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))
