"""HTTP-сервис: демо-окно, стриминговый API и вебхук amoCRM."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .amocrm import MOCK_DEALS, AmoClient, DialogStore, format_note, parse_webhook
from .config import ROOT, Settings
from .kb import load_kb
from .service import SupportService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    deal_id: str | None = None
    # Для вызова без хранилища диалогов: контекст сделки и история передаются прямо в запросе.
    deal: dict | None = None
    history: list[dict] | None = None


class ManagerMessage(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


def create_app(settings: Settings | None = None, service: SupportService | None = None) -> FastAPI:
    settings = settings or Settings()
    kb = load_kb(settings.kb_path)
    service = service or SupportService(settings, kb)
    amo = AmoClient(settings)
    dialogs = DialogStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        warm = asyncio.create_task(service.keep_warm())
        log.info("mode=%s model=%s amo=%s faq=%d rules=%d", service.mode, settings.model,
                 "live" if amo.enabled else "demo", len(kb.faq), len(kb.rules))
        yield
        warm.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await warm
        await amo.close()

    app = FastAPI(title="AI Support Copilot", lifespan=lifespan)
    app.state.service, app.state.dialogs, app.state.amo = service, dialogs, amo

    def resolve(req: AskRequest) -> tuple[dict | None, list[dict]]:
        if req.deal_id:
            if req.deal_id not in MOCK_DEALS:
                raise HTTPException(404, "deal not found")
            history = dialogs.get(req.deal_id)
            dialogs.add(req.deal_id, "client", req.message)
            return MOCK_DEALS[req.deal_id], history
        return req.deal, req.history or []

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/healthz")
    async def health():
        return {"status": "ok", "mode": service.mode}

    @app.get("/api/config")
    async def config():
        return {
            "mode": service.mode,
            "model": settings.model if service.engine else "rules",
            "effort": settings.effort,
            "prompt_cache_ttl": settings.cache_ttl,
            "amo": "live" if amo.enabled else "demo",
            "kb": {"faq": len(kb.faq), "rules": len(kb.rules), "chars": len(kb.text)},
        }

    @app.get("/api/stats")
    async def stats():
        return service.stats

    @app.get("/api/deals")
    async def deals():
        return [{**d, "history": dialogs.get(i)} for i, d in MOCK_DEALS.items()]

    @app.post("/api/deals/{deal_id}/manager")
    async def manager_message(deal_id: str, msg: ManagerMessage):
        if deal_id not in MOCK_DEALS:
            raise HTTPException(404, "deal not found")
        dialogs.add(deal_id, "manager", msg.text)
        return {"history": dialogs.get(deal_id)}

    @app.post("/api/deals/{deal_id}/reset")
    async def reset(deal_id: str):
        dialogs.reset(deal_id)
        return {"ok": True}

    @app.post("/api/ask")
    async def ask(req: AskRequest):
        """Стриминг (SSE): meta → delta… (ответ клиенту по мере генерации) → result (оба блока + метрики)."""
        deal, history = resolve(req)

        async def events():
            async for ev in service.stream(req.message, deal, history):
                yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/answer")
    async def answer(req: AskRequest):
        deal, history = resolve(req)
        return await service.answer(req.message, deal, history)

    async def process_incoming(msg: dict) -> None:
        key = f"{msg['entity_type']}:{msg['entity_id']}"
        history = dialogs.get(key)
        dialogs.add(key, "client", msg["text"])
        try:
            deal = await amo.get_deal(msg["entity_type"], msg["entity_id"])
        except Exception as exc:  # сеть/права amoCRM: отвечаем и без контекста сделки
            log.warning("amo get_deal failed: %s", exc)
            deal = {"id": msg["entity_id"]}
        result = await service.answer(msg["text"], deal, history)
        try:
            await amo.add_note(msg["entity_type"], msg["entity_id"], format_note(result["data"], result["metrics"]))
        except Exception as exc:
            log.warning("amo add_note failed: %s", exc)

    @app.post("/webhook/amocrm")
    async def amocrm_webhook(request: Request, background: BackgroundTasks, secret: str = ""):
        # amoCRM не подписывает вебхуки, поэтому проверяем секрет из URL вебхука.
        if settings.amo_webhook_secret and secret != settings.amo_webhook_secret:
            raise HTTPException(403, "bad secret")
        messages = parse_webhook(await request.body(), request.headers.get("content-type", ""))
        queued = 0
        for msg in messages:
            if not msg["entity_id"]:
                continue
            if msg["direction"] == "outgoing":
                # Ответы менеджера тоже сохраняем: подсказка учитывает, что он уже предлагал.
                dialogs.add(f"{msg['entity_type']}:{msg['entity_id']}", "manager", msg["text"])
                continue
            background.add_task(process_incoming, msg)
            queued += 1
        # amoCRM ждёт быстрый ответ — обработка идёт в фоне.
        return {"ok": True, "queued": queued}

    return app


app = create_app()
