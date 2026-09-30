import json
from urllib.parse import urlencode

from fastapi.testclient import TestClient

from app.amocrm import format_note, parse_webhook
from app.config import Settings
from app.kb import load_kb
from app.llm import ClaudeEngine
from app.main import create_app
from app.service import SupportService

from .conftest import SAMPLE, FakeClient


def sse_events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def make_client(engine_client=None, **settings_kw) -> tuple[TestClient, object]:
    settings = Settings(prewarm=False, **settings_kw)
    kb = load_kb(settings.kb_path)
    engine = ClaudeEngine(settings, kb, client=engine_client) if engine_client else None
    app = create_app(settings, SupportService(settings, kb, engine=engine))
    return TestClient(app), app


def test_config_and_deals_offline_mode():
    client, _ = make_client()
    cfg = client.get("/api/config").json()
    assert cfg["mode"] == "offline" and cfg["amo"] == "demo" and cfg["kb"]["rules"] == 7
    assert len(client.get("/api/deals").json()) == 4
    assert client.get("/").status_code == 200


def test_ask_streams_sse_and_keeps_dialog():
    client, _ = make_client(FakeClient(payload=SAMPLE))
    res = client.post("/api/ask", json={"deal_id": "1002", "message": "Кофе закончился"})
    assert res.headers["content-type"].startswith("text/event-stream")
    events = sse_events(res.text)
    assert events[0]["type"] == "meta" and events[-1]["type"] == "result"
    assert "".join(e["text"] for e in events if e["type"] == "delta") == SAMPLE["reply"]

    client.post("/api/deals/1002/manager", json={"text": "Ответ менеджера"})
    history = next(d for d in client.get("/api/deals").json() if d["id"] == "1002")["history"]
    assert [m["role"] for m in history] == ["client", "manager"]


def test_answer_endpoint_stateless():
    client, _ = make_client()
    res = client.post("/api/answer", json={"message": "Сколько идёт доставка?", "deal": {"client_name": "Олег"}})
    body = res.json()
    assert body["data"]["kb_refs"] == ["Доставка"] and body["metrics"]["source"] == "offline"
    assert body["data"]["reply"].startswith("Олег, здравствуйте")


def test_unknown_deal_404():
    client, _ = make_client()
    assert client.post("/api/ask", json={"deal_id": "nope", "message": "hi"}).status_code == 404


def test_parse_amocrm_webhook_form():
    body = urlencode({
        "account[subdomain]": "demo",
        "message[add][0][text]": "Кофе закончился",
        "message[add][0][type]": "incoming",
        "message[add][0][entity_type]": "lead",
        "message[add][0][entity_id]": "1002",
        "message[add][0][talk_id]": "77",
        "message[add][1][text]": "Добрый день!",
        "message[add][1][type]": "outgoing",
        "message[add][1][element_type]": "2",
        "message[add][1][element_id]": "1002",
    }).encode()
    msgs = parse_webhook(body, "application/x-www-form-urlencoded")
    assert msgs[0] == {"text": "Кофе закончился", "direction": "incoming", "entity_type": "lead",
                       "entity_id": "1002", "talk_id": "77", "author": ""}
    assert msgs[1]["direction"] == "outgoing" and msgs[1]["entity_type"] == "lead"


def test_webhook_processes_incoming_and_checks_secret(caplog):
    client, app = make_client(amo_webhook_secret="s3cret")
    form = urlencode({"message[add][0][text]": "Как часто чистить от накипи?", "message[add][0][type]": "incoming",
                      "message[add][0][entity_type]": "lead", "message[add][0][entity_id]": "1002"})
    headers = {"content-type": "application/x-www-form-urlencoded"}
    assert client.post("/webhook/amocrm?secret=bad", content=form, headers=headers).status_code == 403
    with caplog.at_level("INFO", logger="amocrm"):
        res = client.post("/webhook/amocrm?secret=s3cret", content=form, headers=headers)
    assert res.json() == {"ok": True, "queued": 1}
    note = "\n".join(r.getMessage() for r in caplog.records)
    assert "Черновик ответа клиенту" in note and "Набор для ухода Care Kit" in note
    assert app.state.dialogs.get("lead:1002")[0]["role"] == "client"


def test_format_note_without_upsell():
    note = format_note({"reply": "Ок", "escalate": True, "upsell": {"rule_id": "none", "reason": "Жалоба"}},
                       {"total_ms": 12.3, "source": "offline"})
    assert "Нужен менеджер" in note and "не сейчас — Жалоба" in note
