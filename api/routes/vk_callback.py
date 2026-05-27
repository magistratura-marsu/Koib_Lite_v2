# -*- coding: utf-8 -*-
import logging, secrets
from typing import Optional
from fastapi import APIRouter, Request, BackgroundTasks
import aiohttp
from pydantic import BaseModel
from src.rag_pipeline import RAGPipeline
from src.safety import check_query_safety, check_answer_safety, sanitize_answer
from src.validation import get_blocked_response
from config import VK_CONFIRM_CODE, VK_ACCESS_TOKEN

logger = logging.getLogger("koib.api.vk")
router = APIRouter()
_pipeline: Optional[RAGPipeline] = None

class Message(BaseModel):
    from_id: int
    text: str

class MessageObject(BaseModel):
    message: Message

class VKCallbackPayload(BaseModel):
    type: str
    object: Optional[MessageObject] = None

def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None: _pipeline = RAGPipeline()
    return _pipeline

async def send_vk_message(user_id: int, text: str, session: aiohttp.ClientSession) -> bool:
    if not VK_ACCESS_TOKEN: return False
    url = "https://api.vk.com/method/messages.send"
    params = {"user_id": user_id, "message": text[:4090], "access_token": VK_ACCESS_TOKEN, "v": "5.131", "random_id": secrets.randbits(31)}
    try:
        async with session.post(url, data=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
            if "error" in data: logger.error(f"VK API ошибка: {data['error']}"); return False
            return True
    except Exception as exc: logger.error(f"Ошибка отправки VK: {exc}"); return False

async def _process_vk_request(user_id: int, text: str, session: aiohttp.ClientSession):
    try:
        is_safe, reason = check_query_safety(text)
        if not is_safe:
            await send_vk_message(user_id, "Этот вопрос требует обращения в службу поддержки. Пожалуйста, свяжитесь с нами по горячей линии.", session)
            return

        pipeline = _get_pipeline()
        result = await pipeline.answer(query=text, user_id=str(user_id), k=4, use_memory=True, validate=False)

        answer = result.get("answer", "Не удалось сгенерировать ответ.")
        if result.get("status") == "rejected": answer = get_blocked_response()
            
        is_safe, _ = check_answer_safety(answer)
        if not is_safe: answer = sanitize_answer(answer)

        await send_vk_message(user_id, answer, session)
    except Exception as exc:
        logger.error(f"Ошибка фоновой обработки RAG-пайплайна: {exc}")

@router.post("/vk_callback")
async def vk_webhook(request: Request, background_tasks: BackgroundTasks) -> str:
    try: raw_data = await request.json()
    except Exception: return "ok"

    if raw_data.get("type") == "confirmation": return VK_CONFIRM_CODE

    try: payload = VKCallbackPayload(**raw_data)
    except Exception: return "ok"

    if payload.type == "message_new" and payload.object:
        user_id = payload.object.message.from_id
        text = payload.object.message.text.strip()
        if text:
            session = request.app.state.vk_session
            background_tasks.add_task(_process_vk_request, user_id, text, session)

    return "ok"
