"""Интеграция с amoCRM: вебхук входящих сообщений, контекст сделки, примечание для менеджера.

Без AMO_SUBDOMAIN/AMO_TOKEN работает демо-режим с тестовыми сделками — прототип можно показать
без аккаунта amoCRM, а реальная интеграция включается двумя переменными окружения.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections import defaultdict
from urllib.parse import parse_qsl

import httpx2

from .config import Settings

log = logging.getLogger("amocrm")

MOCK_DEALS: dict[str, dict] = {
    "1001": {
        "id": "1001", "lead_name": "Подбор кофемашины для дома", "client_name": "Анна Смирнова",
        "stage": "Первичный контакт", "budget": 50000, "tags": [], "products": [],
        "notes": "Пришла с сайта, смотрела автоматические машины",
    },
    "1002": {
        "id": "1002", "lead_name": "Повторная покупка", "client_name": "Игорь Петров",
        "stage": "Постпродажное сопровождение", "budget": 39900, "tags": ["кофемашина"],
        "products": ["Кофемашина Aroma One (куплена 2 недели назад)"],
        "notes": "Купил Aroma One, кофе брал разовой пачкой",
    },
    "1003": {
        "id": "1003", "lead_name": "Кофе в офис на 25 человек", "client_name": "Ольга Кузнецова (ООО «Вектор»)",
        "stage": "Переговоры", "budget": 120000, "tags": ["юрлицо"], "products": [],
        "notes": "Офис-менеджер, нужен договор и закрывающие документы",
    },
    "1004": {
        "id": "1004", "lead_name": "Постоянный клиент", "client_name": "Мария Иванова",
        "stage": "Постпродажное сопровождение", "budget": 45900,
        "tags": ["кофемашина", "подписка", "гарантия+"],
        "products": ["Кофемашина Barista Pro 2", "Подписка «Свежая обжарка»", "Гарантия+"],
        "notes": "Лояльный клиент, 2 года с нами",
    },
}


class DialogStore:
    """История диалога по каждой сделке: входящие клиента и ответы менеджера (из UI или вебхуков)."""

    def __init__(self, limit: int = 50):
        self.limit = limit
        self._data: dict[str, list[dict]] = defaultdict(list)

    def get(self, deal_id: str) -> list[dict]:
        return list(self._data.get(deal_id, []))

    def add(self, deal_id: str, role: str, text: str) -> None:
        items = self._data[deal_id]
        items.append({"role": role, "text": text})
        del items[: -self.limit]

    def reset(self, deal_id: str) -> None:
        self._data.pop(deal_id, None)


def parse_webhook(body: bytes, content_type: str = "") -> list[dict]:
    """Разбирает вебхук amoCRM о сообщениях (form-urlencoded вида message[add][0][text]=...).

    Возвращает список {text, direction, entity_type, entity_id, talk_id, author}.
    """
    if "json" in content_type:
        data = json.loads(body or b"{}")
    else:
        data: dict = {}
        for key, value in parse_qsl(body.decode("utf-8", "replace"), keep_blank_values=True):
            parts = re.findall(r"[^\[\]]+", key)
            node = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):
                    break
            else:
                node[parts[-1]] = value

    raw = (data.get("message") or {}).get("add") or {}
    items = raw.values() if isinstance(raw, dict) else raw
    result = []
    for m in items:
        if not isinstance(m, dict) or not m.get("text"):
            continue
        entity_type = str(m.get("entity_type") or m.get("element_type") or "lead").lower()
        entity_type = {"2": "lead", "1": "contact", "leads": "lead", "contacts": "contact"}.get(entity_type, entity_type)
        author = m.get("author") if isinstance(m.get("author"), dict) else {}
        result.append({
            "text": str(m["text"]),
            "direction": "outgoing" if str(m.get("type", "incoming")).lower() == "outgoing" else "incoming",
            "entity_type": entity_type,
            "entity_id": str(m.get("entity_id") or m.get("element_id") or ""),
            "talk_id": str(m.get("talk_id") or ""),
            "author": author.get("name", ""),
        })
    return result


def format_note(data: dict, metrics: dict) -> str:
    up = data.get("upsell") or {}
    lines = ["🤖 ИИ-помощник", "", "💬 Черновик ответа клиенту:", data.get("reply", "")]
    if data.get("escalate"):
        lines += ["", "⚠️ Нужен менеджер: вопрос вне базы или у клиента проблема."]
    if up.get("rule_id") not in (None, "", "none"):
        lines += ["", f"💡 Допродажа: {up.get('product', '')}", f"Почему: {up.get('reason', '')}",
                  f"Что сказать: {up.get('script', '')}"]
    elif up.get("reason"):
        lines += ["", f"💡 Допродажа: не сейчас — {up.get('reason')}"]
    lines += ["", f"⏱ {metrics.get('total_ms', 0):.0f} мс · {metrics.get('source', '')}"]
    return "\n".join(lines)


class AmoClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.amo_enabled
        self._http = (
            httpx2.AsyncClient(
                base_url=f"https://{settings.amo_subdomain}.{settings.amo_domain}",
                headers={"Authorization": f"Bearer {settings.amo_token}"},
                timeout=10.0,
            )
            if self.enabled
            else None
        )

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()

    async def get_deal(self, entity_type: str, entity_id: str) -> dict:
        if not self._http:
            return copy.deepcopy(MOCK_DEALS.get(entity_id, {"id": entity_id}))
        if entity_type != "lead":
            resp = await self._http.get(f"/api/v4/contacts/{entity_id}")
            resp.raise_for_status()
            c = resp.json()
            return {"id": entity_id, "client_name": c.get("name", ""), "tags": _tags(c)}
        resp = await self._http.get(f"/api/v4/leads/{entity_id}", params={"with": "contacts"})
        resp.raise_for_status()
        lead = resp.json()
        return {
            "id": entity_id,
            "lead_name": lead.get("name", ""),
            "client_name": await self._main_contact_name(lead),
            "stage": str(lead.get("status_id", "")),
            "budget": lead.get("price") or 0,
            "tags": _tags(lead),
        }

    async def _main_contact_name(self, lead: dict) -> str:
        contacts = (lead.get("_embedded") or {}).get("contacts") or []
        main = next((c for c in contacts if c.get("is_main")), contacts[0] if contacts else None)
        if not main or not self._http:
            return ""
        resp = await self._http.get(f"/api/v4/contacts/{main['id']}")
        return resp.json().get("name", "") if resp.status_code == 200 else ""

    async def add_note(self, entity_type: str, entity_id: str, text: str) -> None:
        if not self._http:
            log.info("[demo] note for %s %s:\n%s", entity_type, entity_id, text)
            return
        path = "leads" if entity_type == "lead" else "contacts"
        resp = await self._http.post(
            f"/api/v4/{path}/{entity_id}/notes",
            json=[{"note_type": "common", "params": {"text": text}}],
        )
        resp.raise_for_status()


def _tags(entity: dict) -> list[str]:
    return [t.get("name", "").lower() for t in (entity.get("_embedded") or {}).get("tags") or [] if t.get("name")]
