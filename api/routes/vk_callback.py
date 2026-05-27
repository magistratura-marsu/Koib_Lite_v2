# -*- coding: utf-8 -*-
"""
Koib-V-4.6 — VK Callback API (полностью асинхронный)
★ ДОБАВЛЕНО: Rate Limiting (защита от спама)
"""
import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, Any
from fastapi import APIRouter, Request
import aiohttp

from src.generation import AnswerGenerator
from src.safety import check_query_safety, check_answer_safety, sanitize_answer
from src.validation import get_blocked_response
from config import VK_CONFIRM_CODE, VK_ACCESS_TOKEN, VK_GROUP_ID

logger = logging.getLogger("koib.api.vk")
router = APIRouter()
_generator: AnswerGenerator = None

# ★ НОВОЕ: Rate Limiter (скользящее окно)
_user_requests = defaultdict(list)

def _check_rate_limit(user_id: int, limit: int = 5, window: int = 60) -> bool:
    """Защита от спама: максимум `limit` запросов за `window` секунд."""
    now = time.time()
    _user_requests[user_id] = [t for t in _user_requests[user_id] if now - t < window]
    if len(_user_requests[user_id]) >= limit:
        return False
    _user_requests[user_id].append(now)
    return True

def get_generator() -> AnswerGenerator:
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator

async def send_vk_message(user_id: int, text: str) -> bool:
    if not VK_ACCESS_TOKEN:
        logger.warning("VK_ACCESS_TOKEN не задан")
        return False
    url = "https://api.vk.com/method/messages.send"
    params = {
        "user_id": user_id, "message": text, "access_token": VK_ACCESS_TOKEN,
        "v": "5.131", "random_id": abs(hash(f"{user_id}:{text}")) % (2**31),
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=params) as resp:
                data = await resp.json()
                if "error" in data:
                    logger.error(f"VK API ошибка: {data['error']}")
                    return False
                return True
    except Exception as exc:
        logger.error(f"Ошибка отправки VK: {exc}")
        return False

@router.post("/vk_callback")
async def vk_webhook(request: Request) -> Dict[str, Any]:
    data = await request.json()
    if data.get("type") == "confirmation":
        return VK_CONFIRM_CODE
    if data.get("type") == "message_new":
        try:
            msg = data["object"]["message"]
            user_id = msg["from_id"]
            text = msg["text"].strip()
            if not text:
                return "ok"
                
            # ★ НОВОЕ: Блокировка спамеров
            if not _check_rate_limit(user_id):
                await send_vk_message(user_id, "Слишком много запросов. Пожалуйста, подождите минуту.")
                return "ok"

            logger.info(f"Запрос от {user_id}: {text[:100]}")
            is_safe, reason = check_query_safety(text)
            if not is_safe:
                logger.warning(f"Небезопасный запрос от {user_id}: {reason}")
                await send_vk_message(
                    user_id,
                    "Этот вопрос требует обращения в службу поддержки. "
                    "Пожалуйста, свяжитесь с нами по телефону горячей линии."
                )
                return "ok"

            generator = get_generator()
            result = await generator.answer_async(text)
            answer = result.get("answer", "Не удалось сгенерировать ответ.")
            validation = result.get("validation")
            if validation and validation.get("status") == "rejected":
                answer = get_blocked_response()
                
            is_answer_safe, _ = check_answer_safety(answer)
            if not is_answer_safe:
                answer = sanitize_answer(answer)
            if len(answer) > 4096:
                answer = answer[:4090] + "..."
                
            await send_vk_message(user_id, answer)
        except Exception as exc:
            logger.error(f"Ошибка обработки VK: {exc}")
    return "ok"
