import asyncio
import json

import pytest

from app.amocrm import MOCK_DEALS
from app.jsonstream import JsonStringFieldStreamer
from app.llm import ClaudeEngine, Delta, Final
from app.offline import answer_offline
from app.service import SupportService, enforce_upsell_rules

from .conftest import SAMPLE, FakeClient


def run(coro):
    return asyncio.run(coro)


async def collect(agen):
    return [ev async for ev in agen]


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 13, 64, 10_000])
def test_json_streamer_any_chunking(size):
    reply = 'Привет, "мир"!\nЦена: 39 900 ₽ \\ é 😀 конец'
    doc = json.dumps({"reply": reply, "upsell": {"reply": "не это"}}, ensure_ascii=True)
    s = JsonStringFieldStreamer("reply")
    out = "".join(s.feed(doc[i : i + size]) for i in range(0, len(doc), size))
    assert out == reply
    assert s.done


def test_json_streamer_ignores_escaped_key_inside_other_string():
    doc = json.dumps({"note": 'текст "reply": "ложный"', "reply": "настоящий"}, ensure_ascii=False)
    s = JsonStringFieldStreamer("reply")
    assert "".join(s.feed(c) for c in doc) == "настоящий"


def test_kb_parsing(kb):
    titles = [e.title for e in kb.faq]
    assert {"Доставка", "Гарантия", "Кофе в зёрнах"} <= set(titles)
    assert len(kb.rules) == 7
    sub = kb.rule("beans_subscription")
    assert sub.requires == "кофемашина" and sub.exclude == "подписка" and sub.script
    # Строка «Сейчас действует: …» — часть ответа, а не метаданные.
    assert next(e for e in kb.faq if e.title == "Скидки и акции").body.startswith("Сейчас действует")


def test_offline_engine_passes_cases(kb, cases):
    for c in cases:
        r = answer_offline(kb, c["message"], MOCK_DEALS[c["deal_id"]], [])
        assert r["upsell"]["rule_id"] in c["upsell"], c["message"]
        if c["kb"] is not None:
            assert set(r["kb_refs"]) & set(c["kb"]) if c["kb"] else not r["found_in_kb"], c["message"]
        if c["escalate"] is not None:
            assert r["escalate"] == c["escalate"], c["message"]


def test_offline_does_not_repeat_offer_made_by_manager(kb):
    history = [{"role": "manager", "text": "Кстати, есть Подписка «Свежая обжарка» со скидкой"}]
    r = answer_offline(kb, "Кофе закончился, что взять?", MOCK_DEALS["1002"], history)
    assert r["upsell"]["rule_id"] != "beans_subscription"


def test_engine_request_shape_and_streaming(settings, kb):
    client = FakeClient(payload=SAMPLE)
    engine = ClaudeEngine(settings, kb, client=client)
    events = run(collect(engine.stream("Кофе закончился", MOCK_DEALS["1002"], [])))
    deltas = "".join(e.text for e in events if isinstance(e, Delta))
    final = events[-1]
    assert isinstance(final, Final) and final.data == SAMPLE
    assert deltas == SAMPLE["reply"]
    assert final.usage["cache_read_input_tokens"] == 3900

    params = client.beta.messages.calls[0]
    assert params["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "<knowledge_base>" in params["system"][0]["text"]
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["output_config"]["effort"] == "low"
    assert params["fallbacks"] == "default"
    # Изменчивые данные (сделка, диалог) — только в сообщении, системный промпт статичен для кэша.
    assert "Игорь Петров" in params["messages"][0]["content"]
    assert "Игорь Петров" not in params["system"][0]["text"]


def test_engine_haiku_has_no_effort_or_fallbacks(kb):
    from app.config import Settings

    client = FakeClient(payload=SAMPLE)
    engine = ClaudeEngine(Settings(model="claude-haiku-4-5", prewarm=False), kb, client=client)
    run(collect(engine.stream("Привет", None, [])))
    params = client.beta.messages.calls[0]
    assert "effort" not in params["output_config"] and "fallbacks" not in params


def test_prewarm_matches_real_request_prefix(settings, kb):
    client = FakeClient(payload=SAMPLE)
    engine = ClaudeEngine(settings, kb, client=client)
    run(engine.prewarm())
    run(collect(engine.stream("Привет", None, [])))
    warm, real = client.beta.messages.calls
    assert warm["system"] == real["system"] and warm["output_config"] == real["output_config"]
    assert warm["max_tokens"] == 1


def test_service_llm_then_local_cache(settings, kb):
    client = FakeClient(payload=SAMPLE)
    svc = SupportService(settings, kb, engine=ClaudeEngine(settings, kb, client=client))
    first = run(collect(svc.stream("Кофе закончился", MOCK_DEALS["1002"], [])))
    assert [e["type"] for e in first][0] == "meta" and first[-1]["metrics"]["source"] == "llm"
    assert first[-1]["metrics"]["cost_usd"] > 0
    second = run(collect(svc.stream("  кофе ЗАКОНЧИЛСЯ!  ", MOCK_DEALS["1002"], [])))
    assert second[-1]["metrics"]["source"] == "cache"
    assert second[-1]["data"]["reply"] == SAMPLE["reply"]
    assert len(client.beta.messages.calls) == 1


def test_service_falls_back_to_offline_on_api_error(settings, kb):
    client = FakeClient(payload=SAMPLE, fail_after=3)
    svc = SupportService(settings, kb, engine=ClaudeEngine(settings, kb, client=client))
    events = run(collect(svc.stream("Как часто чистить от накипи?", MOCK_DEALS["1002"], [])))
    types = [e["type"] for e in events]
    assert "reset" in types  # клиенту уже ушло начало ответа — UI его очищает
    assert events[-1]["metrics"]["source"] == "offline"
    assert "APIConnectionError" in events[-1]["metrics"]["fallback_reason"]
    assert events[-1]["data"]["upsell"]["rule_id"] == "care_kit"


def test_service_falls_back_on_refusal(settings, kb):
    client = FakeClient(payload=SAMPLE, stop_reason="refusal")
    svc = SupportService(settings, kb, engine=ClaudeEngine(settings, kb, client=client))
    result = run(svc.answer("Кофе закончился", MOCK_DEALS["1002"], []))
    assert result["metrics"]["source"] == "offline"


def test_upsell_guard_filters_owned_and_unknown_products(kb):
    data = json.loads(json.dumps(SAMPLE))
    guarded = enforce_upsell_rules(kb, data, MOCK_DEALS["1004"])  # у клиента уже есть подписка
    assert guarded["upsell"]["rule_id"] == "none"
    data = json.loads(json.dumps(SAMPLE))
    data["upsell"]["rule_id"] = "free_iphone"
    assert enforce_upsell_rules(kb, data, MOCK_DEALS["1002"])["upsell"]["rule_id"] == "none"
    data = json.loads(json.dumps(SAMPLE))
    assert enforce_upsell_rules(kb, data, MOCK_DEALS["1002"])["upsell"]["rule_id"] == "beans_subscription"


def test_bench_runs_end_to_end_with_fake_client(tmp_path):
    import bench

    report = run(bench.main(["--out", str(tmp_path / "r.md")], client=FakeClient(payload=SAMPLE)))
    assert "ours: 1 запрос" in report and "naive: 2 запроса" in report
    assert (tmp_path / "r.md").read_text(encoding="utf-8").startswith("# Бенчмарк")
