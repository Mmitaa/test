"""Настройки сервиса из переменных окружения (и необязательного файла .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Минимальный загрузчик .env: KEY=VALUE, без перезаписи уже заданных переменных."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# Цены за 1 млн токенов, USD (для оценки стоимости запроса в метриках).
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10},
}

# Модели, которые принимают output_config.effort (Haiku 4.5 его не поддерживает).
EFFORT_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1"}
# Модели с серверным fallback при отказе классификатора (fallbacks: "default").
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1"}


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: os.environ.get("LLM_EFFORT", "low"))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("LLM_MAX_TOKENS", "8000")))
    # 1h — кэш базы знаний живёт час между обращениями; 5m — дешевле запись, но чаще холодный старт.
    cache_ttl: str = field(default_factory=lambda: os.environ.get("PROMPT_CACHE_TTL", "1h"))
    prewarm: bool = field(default_factory=lambda: _bool("PROMPT_PREWARM", True))
    request_timeout: float = field(default_factory=lambda: float(os.environ.get("LLM_TIMEOUT", "30")))
    answer_cache_size: int = field(default_factory=lambda: int(os.environ.get("ANSWER_CACHE_SIZE", "512")))
    answer_cache_ttl: float = field(default_factory=lambda: float(os.environ.get("ANSWER_CACHE_TTL", "600")))
    kb_path: Path = field(default_factory=lambda: Path(os.environ.get("KB_PATH", ROOT / "kb" / "knowledge.md")))
    amo_subdomain: str = field(default_factory=lambda: os.environ.get("AMO_SUBDOMAIN", ""))
    amo_domain: str = field(default_factory=lambda: os.environ.get("AMO_DOMAIN", "amocrm.ru"))
    amo_token: str = field(default_factory=lambda: os.environ.get("AMO_TOKEN", ""))
    amo_webhook_secret: str = field(default_factory=lambda: os.environ.get("AMO_WEBHOOK_SECRET", ""))

    @property
    def llm_enabled(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    @property
    def amo_enabled(self) -> bool:
        return bool(self.amo_subdomain and self.amo_token)


def estimate_cost(model: str, usage: dict, cache_ttl: str = "1h") -> float | None:
    price = PRICES.get(model)
    if not price:
        return None
    write_mult = 2.0 if cache_ttl == "1h" else 1.25
    total = (
        usage.get("input_tokens", 0) * price["input"]
        + usage.get("cache_creation_input_tokens", 0) * price["input"] * write_mult
        + usage.get("cache_read_input_tokens", 0) * price["cache_read"]
        + usage.get("output_tokens", 0) * price["output"]
    )
    return round(total / 1_000_000, 6)
