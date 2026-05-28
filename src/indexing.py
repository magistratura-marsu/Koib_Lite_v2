# -*- coding: utf-8 -*-
import json, re, sqlite3, logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from config import (
    INDEX_DIR, DOCSTORE_DIR, EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL, OPENAI_API_KEY,
    BM25_USE_STOPWORDS, BM25_USE_LEMMATIZATION,
    INDEXING_BATCH_SIZE, INDEXING_FLUSH_THRESHOLD, get_device,  # ★ ИЗМЕНЕНО
)

logger = logging.getLogger("koib.indexing")

_GLOBAL_EMBEDDINGS = None

def get_global_embeddings():
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is not None:
        return _GLOBAL_EMBEDDINGS

    if EMBEDDING_PROVIDER == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        device = get_device()  # ★ ИЗМЕНЕНО: автоопределение cuda/cpu
        logger.info(f"Загрузка embedding-модели '{LOCAL_EMBEDDING_MODEL}' на устройстве: {device} "
                    f"(batch_size={INDEXING_BATCH_SIZE})")
        _GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": device},                       # ★ ИЗМЕНЕНО: было "cpu"
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": INDEXING_BATCH_SIZE,                 # ★ ИЗМЕНЕНО: ускоряет в разы на GPU
                "show_progress_bar": True,                         # ★ ИЗМЕНЕНО: видим прогресс
            },
        )
    elif EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        _GLOBAL_EMBEDDINGS = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")

    return _GLOBAL_EMBEDDINGS


RU_STOPWORDS = {"и", "в", "на", "с", "по", "для", "из", "к", "от", "о", "об", "а", "но", "да",
                "не", "что", "как", "это", "то", "же", "бы", "вы", "мы", "он", "она", "они", "оно",
                "я", "ты", "его", "её", "их", "мой", "твой", "наш", "ваш", "свой", "этот", "тот",
                "такой", "который", "весь", "все", "вся", "всё", "быть", "был", "была", "было",
                "были", "будет", "есть", "нет", "ещё", "уже", "только", "если", "или", "при",
                "про", "за", "до", "после", "между", "через", "над", "под", "перед", "так",
                "тоже", "лишь", "ведь", "вот", "даже", "ну", "ли", "ни", "тебя", "мне", "мной",
                "ним", "ней", "нами", "вам", "вас", "нас", "них", "чего", "чему", "чем", "кем",
                "ком", "где", "когда", "зачем", "почему", "куда", "откуда", "какой", "какая", "какие"}

_TOKEN_RE = re.compile(r'[а-яёa-z0-9]+', re.IGNORECASE)
_MORPH_ANALYZER = None


def _get_morph():
    global _MORPH_ANALYZER
    if _MORPH_ANALYZER is None and BM25_USE_LEMMATIZATION:
        try:
            import pymorphy2
            _MORPH_ANALYZER = pymorphy2.MorphAnalyzer()
        except Exception:
            pass
    return _MORPH_ANALYZER


def _lemmatize_token(token: str) -> str:
    morph = _get_morph()
    if morph is None:
        return token
    try:
        return morph.parse(token)[0].normal_form
    except Exception:
        return token


def tokenize_ru(text: str) -> str:
    if not text:
        return ""
    raw_tokens = [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]
    if BM25_USE_STOPWORDS:
        raw_tokens = [t for t in raw_tokens if t not in RU_STOPWORDS]
    tokens = [_lemmatize_token(t) for t in raw_tokens] if BM25_USE_LEMMATIZATION else raw_tokens
    return " ".join(tokens)


def prepare_fts_query(query: str) -> str:
    raw_tokens = [t.lower() for t in _TOKEN_RE.findall(query) if len(t) > 1]
    if BM25_USE_STOPWORDS:
        raw_tokens = [t for t in raw_tokens if t not in RU_STOPWORDS]
    tokens = [_lemmatize_token(t) for t in raw_tokens] if BM25_USE_LEMMATIZATION else raw_tokens
    if not tokens:
        return ""
    seen, unique = set(), []
    for t in tokens[:20]:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return " OR ".join(f'"{t}"' for t in unique)


class DocStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (DOCSTORE_DIR / "docstore.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS docstore "
                "(chunk_id TEXT PRIMARY KEY, content TEXT, chunk_type TEXT, metadata TEXT)"
            )

    def add_many(self, chunks) -> None:
        rows = [
            (c.chunk_id, c.full_content, c.chunk_type, json.dumps(c.metadata, ensure_ascii=False))
            for c in chunks if c.full_content
        ]
        if rows:
            with self.conn:
                self.conn.executemany("INSERT OR REPLACE INTO docstore VALUES (?, ?, ?, ?)", rows)

    def get_content(self, chunk_id: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT content FROM docstore WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
        return row[0] if row else None


class BM25FTSIndex:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (INDEX_DIR / "bm25_fts.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        with self.conn:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
                "chunk_id UNINDEXED, content, chunk_type UNINDEXED, "
                "source UNINDEXED, page UNINDEXED, heading UNINDEXED, "
                "model UNINDEXED, metadata UNINDEXED, "
                "tokenize='unicode61 remove_diacritics 1')"
            )

    def add_chunks(self, chunks) -> None:
        rows = []
        for c in chunks:
            text = c.full_content if c.full_content else c.content
            tokenized = tokenize_ru(text)
            if tokenized:
                rows.append((
                    c.chunk_id, tokenized, c.chunk_type,
                    c.metadata.get("source", ""),
                    str(c.metadata.get("page", 0)),
                    c.metadata.get("heading", ""),
                    c.metadata.get("model", "unknown"),
                    json.dumps(c.metadata, ensure_ascii=False),
                ))
        if rows:
            with self.conn:
                self.conn.executemany("INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def search(self, query: str, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        fts_query = prepare_fts_query(query)
        if not fts_query:
            return []
        try:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT chunk_id, content, chunk_type, source, page, heading, model, metadata, "
                "bm25(chunks_fts) AS rank FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts_query, k),
            )
            results = []
            for row in cur.fetchall():
                metadata = json.loads(row[7]) if row[7] else {}
                metadata.setdefault("chunk_id", row[0])
                metadata.setdefault("content", row[1])
                metadata.setdefault("chunk_type", row[2])
                metadata.setdefault("source", row[3])
                metadata.setdefault("page", int(row[4]) if row[4] else 0)
                metadata.setdefault("heading", row[5])
                score = -float(row[8]) if row[8] is not None else 0.0
                results.append((metadata, score))
            return results
        except Exception:
            return []

    def count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks_fts")
        row = cur.fetchone()
        return row[0] if row else 0


class IndexBuilder:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else INDEX_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_vectorstore = None
        self.summary_vectorstore = None
        self.bm25 = BM25FTSIndex(self.output_dir / "bm25_fts.db")
        self.docstore = DocStore(DOCSTORE_DIR / "docstore.db")
        self._text_docs = []
        self._summary_docs = []

    def add_chunks(self, chunks) -> None:
        self.docstore.add_many(chunks)
        self.bm25.add_chunks(chunks)
        for c in chunks:
            lc_doc = c.to_langchain_doc()
            if c.chunk_type == "text":
                self._text_docs.append(lc_doc)
            else:
                self._summary_docs.append(lc_doc)
        # ★ ИЗМЕНЕНО: порог сброса теперь настраивается через INDEXING_FLUSH_THRESHOLD
        # Для 2 ГБ RAM (VPS): 2000 (защита от OOM)
        # Для 20 ГБ RAM (PC): 50000 (макс. скорость, меньше I/O)
        if len(self._text_docs) + len(self._summary_docs) > INDEXING_FLUSH_THRESHOLD:
            self._flush_vectorstores()

    def _flush_vectorstores(self) -> None:
        if not self._text_docs and not self._summary_docs:
            return
        embeddings = get_global_embeddings()
        try:
            from langchain_community.vectorstores import FAISS
            if self._text_docs:
                if self.text_vectorstore is None:
                    self.text_vectorstore = FAISS.from_documents(self._text_docs, embeddings)
                else:
                    self.text_vectorstore.add_documents(self._text_docs)
                self.text_vectorstore.save_local(str(self.output_dir), index_name="text_index")
                self._text_docs = []
            if self._summary_docs:
                if self.summary_vectorstore is None:
                    self.summary_vectorstore = FAISS.from_documents(self._summary_docs, embeddings)
                else:
                    self.summary_vectorstore.add_documents(self._summary_docs)
                self.summary_vectorstore.save_local(str(self.output_dir), index_name="summary_index")
                self._summary_docs = []
        except Exception as exc:
            logger.error(f"Ошибка сборки FAISS: {exc}")

    def save(self) -> None:
        self._flush_vectorstores()

    def load(self) -> None:
        embeddings = get_global_embeddings()
        try:
            from langchain_community.vectorstores import FAISS
            if (self.output_dir / "text_index.faiss").exists():
                self.text_vectorstore = FAISS.load_local(
                    str(self.output_dir), embeddings,
                    index_name="text_index", allow_dangerous_deserialization=True,
                )
            if (self.output_dir / "summary_index.faiss").exists():
                self.summary_vectorstore = FAISS.load_local(
                    str(self.output_dir), embeddings,
                    index_name="summary_index", allow_dangerous_deserialization=True,
                )
        except Exception:
            pass
