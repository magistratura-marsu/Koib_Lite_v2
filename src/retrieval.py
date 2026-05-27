# -*- coding: utf-8 -*-
"""
Koib-V-4.7 — Модуль гибридного поиска
★ ДОБАВЛЕНО: U-Shape Context Ordering (fix "Lost in the Middle")
★ ДОБАВЛЕНО: обрезка огромных таблиц перед отдачей в генератор
"""
import json
import logging
import sqlite3
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .indexing import IndexBuilder, get_global_embeddings
from config import (
    QUERY_PREFIX, PASSAGE_PREFIX,
    VECTOR_SEARCH_K, BM25_SEARCH_K, FINAL_TOP_K,
    HYBRID_ALPHA, USE_RERANKER, RERANKER_MODEL, USE_ONNX_RERANKER,
    USE_HYDE, EMBEDDING_PROVIDER, METADATA_DIR,
    SEMANTIC_CACHE_ENABLED, SEMANTIC_CACHE_THRESHOLD,
    USE_USHAPED_CONTEXT, MAX_TABLE_ROWS_IN_PROMPT,
    MAX_TABLE_TOKENS_IN_PROMPT,
)
from .utils import estimate_tokens

logger = logging.getLogger("koib.retrieval")


# ═══════════════════════════════════════════════════════════════
# U-Shape Context Ordering (mitigation "Lost in the Middle")
# ═══════════════════════════════════════════════════════════════
def reorder_u_shape(chunks: List) -> List:
    """
    Переставить чанки так, чтобы самые релевантные оказались
    в начале и в конце контекста (LLM лучше "видит" края).

    Вход:  [c1, c2, c3, c4, c5, c6]  (отсортированы по score desc)
    Выход: [c1, c6, c2, c5, c3, c4]
    """
    if len(chunks) <= 2:
        return chunks
    reordered = []
    left, right = 0, len(chunks) - 1
    while left <= right:
        reordered.append(chunks[left])
        if left != right:
            reordered.append(chunks[right])
        left += 1
        right -= 1
    return reordered


# ═══════════════════════════════════════════════════════════════
# Обрезка таблиц (защита от переполнения контекста)
# ═══════════════════════════════════════════════════════════════
def truncate_table_for_prompt(markdown: str,
                              max_rows: int = MAX_TABLE_ROWS_IN_PROMPT,
                              max_tokens: int = MAX_TABLE_TOKENS_IN_PROMPT) -> str:
    """
    Обрезать Markdown-таблицу, если она слишком большая.
    Сохраняет заголовок + разделитель + первые N строк данных.
    """
    if not markdown:
        return markdown
    lines = markdown.split('\n')
    if len(lines) <= max_rows + 2:
        # Проверяем по токенам
        if estimate_tokens(markdown) <= max_tokens:
            return markdown

    # Сохраняем header (строка 0) и separator (строка 1)
    header = lines[0] if lines else ""
    separator = lines[1] if len(lines) > 1 else ""
    data_lines = [l for l in lines[2:] if l.strip()][:max_rows]

    truncated_parts = [header, separator] + data_lines
    truncated = '\n'.join(truncated_parts)

    # Финальная проверка по токенам
    if estimate_tokens(truncated) > max_tokens:
        # Обрезаем построчно
        result_lines = [header, separator]
        current_tokens = estimate_tokens(header + separator)
        for line in data_lines:
            line_tokens = estimate_tokens(line)
            if current_tokens + line_tokens > max_tokens:
                break
            result_lines.append(line)
            current_tokens += line_tokens
        truncated = '\n'.join(result_lines)

    original_rows = max(0, len(lines) - 2)
    kept_rows = len(data_lines)
    if kept_rows < original_rows:
        truncated += f"\n\n[...таблица обрезана: показано {kept_rows} из {original_rows} строк...]"
    return truncated


class SemanticCache:
    def __init__(self, path: Optional[Path] = None,
                 threshold: float = SEMANTIC_CACHE_THRESHOLD):
        self.path = path or METADATA_DIR / "semantic_cache.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    query_text TEXT,
                    embedding BLOB,
                    answer TEXT,
                    sources TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hit_count INTEGER DEFAULT 0
                )
            ''')

    def get(self, query: str, query_embedding: Optional[List[float]]) -> Optional[Dict]:
        if not SEMANTIC_CACHE_ENABLED or query_embedding is None:
            return None
        try:
            cur = self.conn.cursor()
            cur.execute('''
                SELECT query_text, embedding, answer, sources, query_hash
                FROM cache
                ORDER BY hit_count DESC, created_at DESC
                LIMIT 1000
            ''')
            q_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return None
            best_match = None
            best_sim = 0.0
            for row in cur.fetchall():
                cached_emb = np.frombuffer(row[1], dtype=np.float32)
                c_norm = np.linalg.norm(cached_emb)
                if c_norm == 0:
                    continue
                sim = float(np.dot(q_vec, cached_emb) / (q_norm * c_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_match = (row, sim)
            if best_match and best_match[1] >= self.threshold:
                row, sim = best_match
                self.conn.execute(
                    'UPDATE cache SET hit_count = hit_count + 1 WHERE query_hash = ?',
                    (row[4],)
                )
                self.conn.commit()
                logger.info(f"Semantic cache HIT (sim={sim:.3f})")
                return {
                    "answer": row[2],
                    "sources": json.loads(row[3]) if row[3] else [],
                    "similarity": sim,
                }
        except Exception as exc:
            logger.debug(f"Semantic cache get error: {exc}")
        return None

    def set(self, query: str, query_embedding: Optional[List[float]],
            answer: str, sources: List[Dict]) -> None:
        if not SEMANTIC_CACHE_ENABLED or query_embedding is None:
            return
        try:
            q_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
            emb_blob = np.array(query_embedding, dtype=np.float32).tobytes()
            with self.conn:
                self.conn.execute(
                    'INSERT OR REPLACE INTO cache '
                    '(query_hash, query_text, embedding, answer, sources) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (q_hash, query, emb_blob, answer,
                     json.dumps(sources, ensure_ascii=False)),
                )
        except Exception as exc:
            logger.debug(f"Semantic cache set error: {exc}")

    def clear(self) -> None:
        try:
            with self.conn:
                self.conn.execute('DELETE FROM cache')
        except Exception as exc:
            logger.debug(f"Semantic cache clear error: {exc}")


class ResponseCache:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or METADATA_DIR / "response_cache.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    hypothetical TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute('SELECT hypothetical FROM cache WHERE query_hash = ?',
                    (self._hash(query),))
        row = cur.fetchone()
        return row[0] if row else None

    def set(self, query: str, hypothetical: str) -> None:
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO cache (query_hash, hypothetical) VALUES (?, ?)',
                (self._hash(query), hypothetical),
            )

    def clear(self) -> None:
        try:
            with self.conn:
                self.conn.execute('DELETE FROM cache')
        except Exception as exc:
            logger.debug(f"ResponseCache clear error: {exc}")


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    full_content: Optional[str] = None
    score: float = 0.0
    source: str = ""
    page: int = 0
    heading: str = ""
    model: str = "unknown"
    chunk_type: str = "text"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_context_string(self) -> str:
        """
        Сформировать строку для вставки в промпт.
        ★ Таблицы обрезаются для защиты от переполнения контекста.
        """
        parts = [f"[Документ: {self.source}, стр. {self.page}]"]
        if self.heading:
            parts.append(f"Раздел: {self.heading}")

        display_content = self.full_content or self.content

        if self.chunk_type == "table":
            # ★ ОБРЕЗКА больших таблиц
            display_content = truncate_table_for_prompt(display_content)
            parts.append(f"ТАБЛИЦА:\n{display_content}")
        elif self.chunk_type == "formula":
            parts.append(f"ФОРМУЛА: {display_content}")
        elif self.chunk_type == "figure":
            parts.append(f"РИСУНОК: {display_content}")
        else:
            parts.append(display_content)
        return "\n".join(parts)


TABLE_KEYWORDS = {"таблиц", "значени", "параметр", "сводк", "данные", "показател"}
FORMULA_KEYWORDS = {"формул", "вычислен", "расчёт", "уравнен", "коэффициент"}
FIGURE_KEYWORDS = {"схем", "рисунок", "диаграмм", "чертёж", "график"}


def _detect_query_intent(query: str) -> Dict[str, float]:
    query_lower = query.lower()
    intent = {"table": 0.0, "formula": 0.0, "figure": 0.0, "text": 1.0}
    table_hits = sum(1 for kw in TABLE_KEYWORDS if kw in query_lower)
    formula_hits = sum(1 for kw in FORMULA_KEYWORDS if kw in query_lower)
    figure_hits = sum(1 for kw in FIGURE_KEYWORDS if kw in query_lower)
    total_hits = table_hits + formula_hits + figure_hits
    if total_hits > 0:
        intent["table"] = min(table_hits / 2.0, 1.0)
        intent["formula"] = min(formula_hits / 2.0, 1.0)
        intent["figure"] = min(figure_hits / 2.0, 1.0)
        intent["text"] = max(0.3, 1.0 - total_hits * 0.2)
    return intent


class HybridRetriever:
    def __init__(self, index_builder: Optional[IndexBuilder] = None):
        self.index_builder = index_builder or IndexBuilder()
        self.index_builder.load()
        self._reranker = None
        self._cache = ResponseCache()
        self._semantic_cache = SemanticCache()

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        if not USE_RERANKER:
            return None
        try:
            if USE_ONNX_RERANKER:
                try:
                    from sentence_transformers import CrossEncoder
                    self._reranker = CrossEncoder(RERANKER_MODEL, backend="onnx")
                    return self._reranker
                except Exception as onnx_exc:
                    logger.warning(f"ONNX fallback: {onnx_exc}")
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_MODEL)
            return self._reranker
        except Exception as exc:
            logger.warning(f"Не удалось загрузить реранжер: {exc}")
            return None

    def search(self, query: str, k: int = FINAL_TOP_K,
               model_filter: str = "", use_hyde: Optional[bool] = None) -> List[RetrievalResult]:
        query_embedding = None
        try:
            embeddings = get_global_embeddings()
            if hasattr(embeddings, 'embed_query'):
                query_embedding = embeddings.embed_query(
                    (QUERY_PREFIX + query) if EMBEDDING_PROVIDER == "local" else query
                )
        except Exception:
            pass

        cached = self._semantic_cache.get(query, query_embedding)
        if cached:
            logger.info(f"Ответ из семантического кэша (sim={cached['similarity']:.3f})")
            return []

        intent = _detect_query_intent(query)
        search_query = query
        use_hyde_flag = use_hyde if use_hyde is not None else USE_HYDE
        if use_hyde_flag:
            hyde_result = self._apply_hyde(query)
            if hyde_result:
                search_query = hyde_result

        vector_results = self._vector_search(search_query, intent, model_filter)
        bm25_results = self._bm25_search(query, model_filter)
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        try:
            from .quarantine import filter_quarantined_chunks
            fused = filter_quarantined_chunks(fused)
        except Exception:
            pass

        if USE_RERANKER and len(fused) > k:
            reranker = self._get_reranker()
            if reranker:
                fused = self._rerank(query, fused, reranker)

        results = fused[:k]

        # Подгружаем full_content из DocStore для структурированных чанков
        for r in results:
            if r.chunk_type in ("table", "formula", "figure") and r.full_content is None:
                full = self.index_builder.docstore.get_content(r.chunk_id)
                if full:
                    r.full_content = full

        # ★ U-Shape reordering для mitigation "Lost in the Middle"
        if USE_USHAPED_CONTEXT and len(results) > 2:
            results = reorder_u_shape(results)
            logger.debug(f"Применён U-Shape reordering для {len(results)} чанков")

        return results

    def _vector_search(self, query: str, intent: Dict[str, float],
                       model_filter: str = "") -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        seen_ids: set = set()
        search_text = f"{QUERY_PREFIX}{query}" if EMBEDDING_PROVIDER == "local" else query

        if self.index_builder.text_vectorstore is not None:
            try:
                k_text = int(VECTOR_SEARCH_K * intent.get("text", 1.0)) + 3
                docs = self.index_builder.text_vectorstore.similarity_search_with_score(search_text, k=k_text)
                for doc, score in docs:
                    chunk_id = doc.metadata.get("chunk_id", "")
                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)
                    doc_model = doc.metadata.get("model", "unknown")
                    if model_filter and doc_model != "unknown" and doc_model != model_filter:
                        continue
                    results.append(RetrievalResult(
                        chunk_id=chunk_id, content=doc.page_content, score=float(score),
                        source=doc.metadata.get("source", ""), page=doc.metadata.get("page", 0),
                        heading=doc.metadata.get("heading", ""), model=doc_model,
                        chunk_type=doc.metadata.get("chunk_type", "text"), metadata=doc.metadata,
                    ))
            except Exception as exc:
                logger.warning(f"Ошибка векторного поиска по текстам: {exc}")

        if self.index_builder.summary_vectorstore is not None:
            try:
                k_struct = int(VECTOR_SEARCH_K * max(intent["table"], intent["formula"], 0.3)) + 3
                docs = self.index_builder.summary_vectorstore.similarity_search_with_score(search_text, k=k_struct)
                for doc, score in docs:
                    chunk_id = doc.metadata.get("chunk_id", "")
                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)
                    doc_model = doc.metadata.get("model", "unknown")
                    if model_filter and doc_model != "unknown" and doc_model != model_filter:
                        continue
                    real_chunk_type = doc.metadata.get("chunk_type", "text")
                    if real_chunk_type not in ("table", "formula", "figure"):
                        real_chunk_type = "text"
                    results.append(RetrievalResult(
                        chunk_id=chunk_id, content=doc.page_content, score=float(score),
                        source=doc.metadata.get("source", ""), page=doc.metadata.get("page", 0),
                        heading=doc.metadata.get("heading", ""), model=doc_model,
                        chunk_type=real_chunk_type, metadata=doc.metadata,
                    ))
            except Exception as exc:
                logger.warning(f"Ошибка векторного поиска по сводкам: {exc}")

        return results

    def _bm25_search(self, query: str, model_filter: str = "") -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        bm25_hits = self.index_builder.bm25.search(query, k=BM25_SEARCH_K)
        for metadata, score in bm25_hits:
            doc_model = metadata.get("model", "unknown")
            if model_filter and doc_model != "unknown" and doc_model != model_filter:
                continue
            results.append(RetrievalResult(
                chunk_id=metadata.get("chunk_id", ""), content=metadata.get("content", ""),
                score=score, source=metadata.get("source", ""), page=metadata.get("page", 0),
                heading=metadata.get("heading", ""), model=doc_model,
                chunk_type=metadata.get("chunk_type", "text"), metadata=metadata,
            ))
        return results

    def _reciprocal_rank_fusion(self, vector_results: List[RetrievalResult],
                                bm25_results: List[RetrievalResult], k_rrf: int = 60) -> List[RetrievalResult]:
        chunk_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievalResult] = {}
        for rank, r in enumerate(vector_results, 1):
            chunk_scores.setdefault(r.chunk_id, 0.0)
            chunk_map[r.chunk_id] = r
            chunk_scores[r.chunk_id] += HYBRID_ALPHA / (k_rrf + rank)
        for rank, r in enumerate(bm25_results, 1):
            chunk_scores.setdefault(r.chunk_id, 0.0)
            chunk_map[r.chunk_id] = r
            chunk_scores[r.chunk_id] += (1 - HYBRID_ALPHA) / (k_rrf + rank)
        sorted_ids = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)
        results = []
        for cid in sorted_ids:
            r = chunk_map[cid]
            r.score = chunk_scores[cid]
            results.append(r)
        return results

    def _rerank(self, query: str, results: List[RetrievalResult], reranker) -> List[RetrievalResult]:
        try:
            pairs = [(query, r.content) for r in results]
            scores = reranker.predict(pairs)
            for r, score in zip(results, scores):
                r.score = float(score)
            results.sort(key=lambda x: x.score, reverse=True)
            return results
        except Exception as exc:
            logger.warning(f"Ошибка переранжирования: {exc}")
            return results

    def _apply_hyde(self, query: str) -> Optional[str]:
        cached = self._cache.get(query)
        if cached:
            return cached
        try:
            from .generation import LLMClient
            client = LLMClient()
            hypothetical = client.generate(
                f"Ответь кратко на вопрос, как если бы ты был экспертом "
                f"по технической документации:\n{query}", max_tokens=300,
            )
            if hypothetical and len(hypothetical) > 20:
                self._cache.set(query, hypothetical)
                return hypothetical
        except Exception as exc:
            logger.debug(f"HyDE ошибка: {exc}")
        return None
