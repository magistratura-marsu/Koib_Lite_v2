# -*- coding: utf-8 -*-
import re
import uuid
import hashlib
import logging
import sqlite3
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from config import METADATA_DIR

logger = logging.getLogger("koib.utils")

# ... (clean_text, text_hash, estimate_tokens, truncate_to_tokens, generate_unique_id, 
# detect_model_in_text, detect_model_from_filename, find_figure_caption, extract_headings 
# остаются без изменений из предыдущих версий) ...

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\w\s\-\+\=\*\/\(\)\[\]\{\}\$\<\>\,\.\;\:\!\?\%\&\|\^\~`\"\'\\@\#№°±≥≤≈×÷→←↑↓∈∑∫∂∇∞≈≠√∏∝∧∨¬⊂⊃⊆⊇∅∩∪\u0400-\u04FF\u2116\n\r\t]', '', text, flags=re.UNICODE)
    lines = [line.strip() for line in text.split('\n')]
    while lines and not lines[0]: lines.pop(0)
    while lines and not lines[-1]: lines.pop()
    return '\n'.join(lines)

def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def estimate_tokens(text: str) -> int:
    if not text: return 0
    return max(1, int(len(text) * 0.4))

def generate_unique_id(prefix: str = "") -> str:
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}{uid}" if prefix else uid

# ═══════════════════════════════════════════════════════════════
# Память диалога и Query Rewriting
# ═══════════════════════════════════════════════════════════════
class ConversationMemory:
    def __init__(self, db_path: Optional[Path] = None, max_history: int = 5):
        self.db_path = db_path or (METADATA_DIR / "conversation_memory.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS history (
                user_id TEXT, role TEXT, content TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            self.conn.execute('''CREATE INDEX IF NOT EXISTS idx_user_timestamp ON history(user_id, timestamp)''')

    async def add_message(self, user_id: str, role: str, content: str):
        await asyncio.to_thread(self._sync_add, user_id, role, content)

    def _sync_add(self, user_id: str, role: str, content: str):
        with self.conn:
            self.conn.execute('INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)', (user_id, role, content))

    async def get_history(self, user_id: str) -> List[Dict[str, str]]:
        return await asyncio.to_thread(self._sync_get, user_id)

    def _sync_get(self, user_id: str) -> List[Dict[str, str]]:
        cur = self.conn.cursor()
        cur.execute('SELECT role, content FROM history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (user_id, self.max_history))
        return [{'role': r[0], 'content': r[1]} for r in reversed(cur.fetchall())]

QUERY_REWRITE_PROMPT = """История диалога:
{history}

Текущий вопрос пользователя:
{query}

Переформулируй текущий вопрос в самостоятельный запрос для поиска по технической документации.
Раскрой все местоимения ("она", "его", "этот параметр") на основе контекста диалога.
Верни ТОЛЬКО переформулированный вопрос, без пояснений."""

async def rewrite_query(query: str, history: List[Dict[str, str]], llm_client) -> str:
    if not history or len(history) < 2: return query
    history_text = '\n'.join(f"{m['role'].capitalize()}: {m['content'][:200]}" for m in history[-4:])
    prompt = QUERY_REWRITE_PROMPT.format(history=history_text, query=query)
    try:
        rewritten = await llm_client.generate_async(prompt, max_tokens=150, temperature=0.01)
        rewritten = rewritten.strip()
        if 10 < len(rewritten) < 500: return rewritten
    except Exception: pass
    return query
