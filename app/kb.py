"""Загрузка базы знаний из Markdown.

Весь файл целиком уходит в системный промпт (он короткий и кэшируется на стороне API),
а разобранные записи используются резервным движком без ИИ и для проверки ответов модели.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_WORD_RE = re.compile(r"[a-zа-я0-9+]+")
STEM_LEN = 5


def normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower().replace("ё", "е")))


def stems(text: str) -> set[str]:
    """Грубый стемминг для русского: первые 5 букв слова длиннее 2 символов."""
    return {w[:STEM_LEN] for w in normalize(text).split() if len(w) > 2}


def phrase_stems(phrases: list[str]) -> list[tuple[str, ...]]:
    return [tuple(w[:STEM_LEN] for w in normalize(p).split() if len(w) > 2) for p in phrases]


@dataclass
class FaqEntry:
    title: str
    body: str
    keywords: list[str]

    def score(self, question_stems: set[str]) -> int:
        score = 0
        for phrase in phrase_stems(self.keywords + [self.title]):
            if phrase and all(s in question_stems for s in phrase):
                score += len(phrase)
        return score


@dataclass
class UpsellRule:
    id: str
    title: str
    product: str
    triggers: list[str]
    requires: str
    exclude: str
    why: str
    script: str

    def trigger_score(self, question_stems: set[str]) -> int:
        return sum(
            len(p) for p in phrase_stems(self.triggers) if p and all(s in question_stems for s in p)
        )


@dataclass
class KnowledgeBase:
    text: str
    faq: list[FaqEntry] = field(default_factory=list)
    rules: list[UpsellRule] = field(default_factory=list)

    def rule(self, rule_id: str) -> UpsellRule | None:
        return next((r for r in self.rules if r.id == rule_id), None)


def _split_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


_META_KEYS = {"ключи", "id", "товар", "триггеры", "нужно", "исключить", "почему", "фраза"}


def _parse_entry(lines: list[str]) -> tuple[dict[str, str], str]:
    meta: dict[str, str] = {}
    body: list[str] = []
    for line in lines:
        key, sep, value = line.strip().partition(":")
        if sep and not body and key.strip().lower() in _META_KEYS:
            meta[key.strip().lower()] = value.strip()
        elif line.strip():
            body.append(line.strip())
    return meta, "\n".join(body)


def load_kb(path: Path) -> KnowledgeBase:
    text = path.read_text(encoding="utf-8").strip()
    kb = KnowledgeBase(text=text)
    section = ""
    title: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if title is None:
            return
        meta, body = _parse_entry(buf)
        if "допродаж" in section.lower() and "id" in meta:
            kb.rules.append(
                UpsellRule(
                    id=meta["id"],
                    title=title,
                    product=meta.get("товар", title),
                    triggers=_split_list(meta.get("триггеры", "")),
                    requires=meta.get("нужно", "").strip().lower(),
                    exclude=meta.get("исключить", "").strip().lower(),
                    why=meta.get("почему", ""),
                    script=meta.get("фраза", ""),
                )
            )
        elif "вопрос" in section.lower():
            kb.faq.append(FaqEntry(title=title, body=body, keywords=_split_list(meta.get("ключи", ""))))

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            section, title, buf = line[3:].strip(), None, []
        elif line.startswith("### "):
            flush()
            title, buf = line[4:].strip(), []
        elif title is not None:
            buf.append(line)
    flush()
    return kb
