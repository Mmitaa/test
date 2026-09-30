"""Бенчмарк: наш конвейер против «типичного» решения на одних и тех же обращениях.

ours  — 1 запрос к Claude, стриминг, JSON-схема, база знаний в кэше промпта (как в сервисе).
naive — типичная сборка: 2 последовательных запроса (ответ, затем допродажа), без стриминга и без кэша.

Запуск:  ANTHROPIC_API_KEY=... python bench.py [--runs 2] [--model claude-opus-5-5]
Результат печатается и сохраняется в bench/results.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import anthropic

from app.amocrm import MOCK_DEALS
from app.config import ROOT, Settings, estimate_cost
from app.kb import load_kb
from app.llm import ClaudeEngine
from app.prompt import build_user_content
from app.service import SupportService

NAIVE_REPLY = (
    "Ты — оператор поддержки магазина. Ответь клиенту вежливо, на «вы», 2–5 предложений, "
    "только по фактам из базы знаний. Если ответа нет — скажи, что уточнишь.\n\n{kb}"
)
NAIVE_UPSELL = (
    "Ты помогаешь менеджеру с допродажами. По базе знаний, сделке и вопросу клиента выбери одно правило "
    "из раздела «Правила допродаж». Первой строкой напиши только его id (или none), затем почему и что сказать.\n\n{kb}"
)


def check_quality(case: dict, data: dict) -> dict:
    up = (data.get("upsell") or {}).get("rule_id", "none")
    kb_ok = True
    if case["kb"] is not None:
        kb_ok = bool(set(data.get("kb_refs") or []) & set(case["kb"])) if case["kb"] else not data.get("found_in_kb")
    esc_ok = case["escalate"] is None or data.get("escalate") == case["escalate"]
    reply = (data.get("reply") or "").lower()
    safe = not any(f.lower() in reply for f in case.get("forbidden", []))
    return {"kb": kb_ok, "upsell": up in case["upsell"], "escalate": esc_ok, "safe": safe}


async def run_ours(settings: Settings, cases: list[dict], runs: int, client=None) -> list[dict]:
    kb = load_kb(settings.kb_path)
    engine = ClaudeEngine(settings, kb, client=client)
    svc = SupportService(settings, kb, engine=engine)
    await engine.prewarm()  # как в проде: кэш промпта прогревается при старте сервиса
    rows = []
    for r in range(runs):
        for case in cases:
            deal = MOCK_DEALS[case["deal_id"]]
            # Непустая история отключает локальный кэш ответов — меряем именно путь через модель.
            result = await svc.answer(case["message"], deal, [{"role": "manager", "text": "Здравствуйте!"}])
            m = result["metrics"]
            rows.append({"run": r, "case": case["message"], "ttft": m["ttft_ms"], "total": m["total_ms"],
                         "cost": m["cost_usd"] or 0, "cache_read": m["usage"].get("cache_read_input_tokens", 0),
                         "source": m["source"], "quality": check_quality(case, result["data"])})
    return rows


async def run_naive(settings: Settings, cases: list[dict], runs: int, client=None) -> list[dict]:
    kb = load_kb(settings.kb_path)
    client = client or anthropic.AsyncAnthropic()
    rows = []
    for r in range(runs):
        for case in cases:
            deal = MOCK_DEALS[case["deal_id"]]
            content = build_user_content(case["message"], deal, [])
            t0 = time.perf_counter()
            first = await client.beta.messages.create(
                model=settings.model, max_tokens=settings.max_tokens, system=NAIVE_REPLY.format(kb=kb.text),
                messages=[{"role": "user", "content": content}])
            reply_ms = (time.perf_counter() - t0) * 1000
            reply = "".join(b.text for b in first.content if b.type == "text")
            second = await client.beta.messages.create(
                model=settings.model, max_tokens=settings.max_tokens, system=NAIVE_UPSELL.format(kb=kb.text),
                messages=[{"role": "user", "content": f"{content}\n<reply>{reply}</reply>"}])
            total_ms = (time.perf_counter() - t0) * 1000
            text = "".join(b.text for b in second.content if b.type == "text").strip()
            rule_id = text.splitlines()[0].strip().strip("`*\"' ") if text else "none"
            cost = sum(estimate_cost(settings.model, _usage(m.usage)) or 0 for m in (first, second))
            rows.append({"run": r, "case": case["message"], "ttft": reply_ms, "total": total_ms, "cost": cost,
                         "cache_read": 0, "source": "llm",
                         "quality": {"upsell": rule_id in case["upsell"]}})
    return rows


def _usage(u) -> dict:
    return {k: getattr(u, k, 0) or 0 for k in
            ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}


def p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20)[-1] if len(values) >= 2 else values[0]


def summarize(name: str, rows: list[dict]) -> dict:
    ttft = [r["ttft"] for r in rows]
    total = [r["total"] for r in rows]
    q = [r["quality"] for r in rows]
    upsell_ok = sum(x["upsell"] for x in q) / len(q)
    full_ok = [all(x.values()) for x in q if len(x) > 1]
    return {
        "name": name, "n": len(rows),
        "ttft_p50": statistics.median(ttft), "ttft_p95": p95(ttft),
        "total_p50": statistics.median(total), "total_p95": p95(total),
        "cost_1000": sum(r["cost"] for r in rows) / len(rows) * 1000,
        "upsell_ok": upsell_ok, "all_ok": (sum(full_ok) / len(full_ok)) if full_ok else None,
        "llm_calls": 1 if name.startswith("ours") else 2,
        "fallbacks": sum(r["source"] != "llm" for r in rows),
    }


def render(summaries: list[dict], model: str) -> str:
    fmt = lambda v: f"{v / 1000:.2f} с" if v >= 1000 else f"{v:.0f} мс"  # noqa: E731
    pct = lambda v: "—" if v is None else f"{v * 100:.0f}%"  # noqa: E731
    lines = [
        f"# Бенчмарк · модель `{model}`", "",
        "| Подход | Запросов к LLM | Первое слово p50 / p95 | Полный ответ p50 / p95 | $ на 1000 обращений | Допродажа верно | Все проверки |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['name']} | {s['llm_calls']} | {fmt(s['ttft_p50'])} / {fmt(s['ttft_p95'])} | "
            f"{fmt(s['total_p50'])} / {fmt(s['total_p95'])} | ${s['cost_1000']:.2f} | {pct(s['upsell_ok'])} | {pct(s['all_ok'])} |"
        )
    lines += ["", f"Обращений в прогоне: {summaries[0]['n']}. Уходов в резервный режим: "
              + ", ".join(f"{s['name']} — {s['fallbacks']}" for s in summaries)]
    return "\n".join(lines)


async def main(argv: list[str] | None = None, client=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=["both", "ours", "naive"], default="both")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results.md"))
    args = ap.parse_args(argv)

    settings = Settings(**({"model": args.model} if args.model else {}), prewarm=True)
    if client is None and not settings.llm_enabled:
        sys.exit("Нужен ANTHROPIC_API_KEY: бенчмарк делает реальные запросы к Claude.")
    cases = json.loads((ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))

    summaries = []
    if args.mode in ("both", "ours"):
        summaries.append(summarize("ours: 1 запрос + стриминг + кэш", await run_ours(settings, cases, args.runs, client)))
    if args.mode in ("both", "naive"):
        summaries.append(summarize("naive: 2 запроса, без кэша", await run_naive(settings, cases, args.runs, client)))
    report = render(summaries, settings.model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n", encoding="utf-8")
    print(report)
    return report


if __name__ == "__main__":
    asyncio.run(main())
