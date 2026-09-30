"""Резервный движок без ИИ: поиск по базе знаний + правила допродаж.

Работает, когда нет ключа API или Claude недоступен — сервис не падает, а отвечает по базе.
"""

from __future__ import annotations

from .kb import KnowledgeBase, normalize, stems

_NEGATIVE = (
    "ужас", "недовол", "разочар", "обман", "жалоб", "позор", "хамств", "безобраз",
    "плохо", "отврат", "кошмар", "верните деньги",
)
# Проблемы, при которых допродажа неуместна, а обращение нужно передать человеку.
_PROBLEM = ("сломал", "не работает", "брак", "возврат", "вернуть", "задерживает", "не пришел", "неисправ")
_POSITIVE = ("спасибо", "отлично", "класс", "супер", "благодар", "нравится", "доволен", "понравил")

FALLBACK_REPLY = (
    "Спасибо за вопрос! Хочу ответить точно, поэтому уточню информацию у коллег — "
    "менеджер вернётся к вам с ответом в ближайшее время."
)


def _greeting(name: str | None) -> str:
    return f"{name}, здравствуйте! " if name else "Здравствуйте! "


def _first_name(deal: dict) -> str | None:
    name = (deal.get("client_name") or "").strip()
    return name.split()[0] if name else None


def answer_offline(kb: KnowledgeBase, message: str, deal: dict | None, history: list[dict]) -> dict:
    deal = deal or {}
    q = stems(message)
    text = normalize(message)
    tags = {t.lower() for t in deal.get("tags", [])}

    negative = any(w in text for w in _NEGATIVE)
    problem = negative or any(w in text for w in _PROBLEM)
    sentiment = "negative" if negative else ("positive" if any(w in text for w in _POSITIVE) else "neutral")

    scored = sorted(((e.score(q), i, e) for i, e in enumerate(kb.faq)), key=lambda x: (-x[0], x[1]))
    best = scored[0][2] if scored and scored[0][0] > 0 else None

    greet = _greeting(_first_name(deal)) if not history else ""
    if best:
        apology = "Очень жаль, что так вышло, обязательно разберёмся. " if problem else ""
        reply = f"{greet}{apology}{best.body.splitlines()[0]}"
        extra = best.body.splitlines()[1:2]
        if extra:
            reply += f" {extra[0]}"
    else:
        reply = f"{greet}{FALLBACK_REPLY}"

    upsell = _pick_upsell(kb, q, tags, history) if not problem else None
    if upsell:
        up = {
            "rule_id": upsell.id,
            "product": upsell.product,
            "reason": upsell.why[:1].upper() + upsell.why[1:] + ".",
            "script": upsell.script,
            "priority": "high" if upsell.trigger_score(q) >= 2 else "medium",
        }
    else:
        reason = (
            "У клиента проблема — сначала решите её, допродажа сейчас навредит."
            if problem
            else "Явного повода для допродажи нет — сначала ответьте на вопрос клиента."
        )
        up = {"rule_id": "none", "product": "", "reason": reason, "script": "", "priority": "none"}

    return {
        "reply": reply,
        "found_in_kb": best is not None,
        "escalate": best is None or problem,
        "sentiment": sentiment,
        "kb_refs": [best.title] if best else [],
        "upsell": up,
    }


def _pick_upsell(kb: KnowledgeBase, q: set[str], tags: set[str], history: list[dict]):
    offered = " ".join(m.get("text", "") for m in history if m.get("role") == "manager").lower()
    best, best_score = None, 0
    for rule in kb.rules:
        if rule.exclude and rule.exclude in tags:
            continue
        if rule.requires and rule.requires not in tags:
            continue
        # Не повторяем то, что менеджер уже предлагал в этом диалоге.
        if rule.product.split("—")[0].strip().lower()[:12] in offered:
            continue
        score = rule.trigger_score(q)
        if score > best_score:
            best, best_score = rule, score
    return best
