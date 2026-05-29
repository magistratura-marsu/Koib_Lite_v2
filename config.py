# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
load_dotenv()  # Принудительно загружает переменные из .env
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = Path(os.getenv("KOIB_DOCS_DIR", str(DATA_DIR / "docs")))
OUTPUT_DIR = Path(os.getenv("KOIB_OUTPUT_DIR", str(BASE_DIR / "output")))
INDEX_DIR = OUTPUT_DIR / "index"
DOCSTORE_DIR = OUTPUT_DIR / "docstore"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
METADATA_DIR = OUTPUT_DIR / "metadata"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gigachat")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TEXT_CHUNK_SIZE = int(os.getenv("TEXT_CHUNK_SIZE", "800"))
TEXT_CHUNK_OVERLAP = int(os.getenv("TEXT_CHUNK_OVERLAP", "80"))
MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "50"))

VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "15"))
BM25_SEARCH_K = int(os.getenv("BM25_SEARCH_K", "10"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "4"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))

USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_ONNX_RERANKER = os.getenv("USE_ONNX_RERANKER", "true").lower() == "true"
USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"
BM25_USE_STOPWORDS = os.getenv("BM25_USE_STOPWORDS", "true").lower() == "true"
BM25_USE_LEMMATIZATION = os.getenv("BM25_USE_LEMMATIZATION", "true").lower() == "true"

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
GIGACHAT_TEMPERATURE = float(os.getenv("GIGACHAT_TEMPERATURE", "0.2"))
GIGACHAT_MAX_TOKENS = int(os.getenv("GIGACHAT_MAX_TOKENS", "1536"))
GIGACHAT_TIMEOUT = int(os.getenv("GIGACHAT_TIMEOUT", "45"))
GIGACHAT_VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true"

OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1536"))

LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "IlyaGusev/saiga_mistral_7b")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")

VALIDATION_IGNORE_QUOTES = os.getenv("VALIDATION_IGNORE_QUOTES", "true").lower() == "true"
UNCERTAINTY_MIN_LENGTH = int(os.getenv("UNCERTAINTY_MIN_LENGTH", "50"))
VALIDATION_USE_LLM_JUDGE = os.getenv("VALIDATION_USE_LLM_JUDGE", "false").lower() == "true"
VALIDATION_CHECK_CITATIONS = os.getenv("VALIDATION_CHECK_CITATIONS", "true").lower() == "true"

OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "50"))
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "80"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "80"))
PARSING_ENGINE = os.getenv("PARSING_ENGINE", "pymupdf")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

VK_CONFIRM_CODE = os.getenv("VK_CONFIRM_CODE", "12345678")
VK_GROUP_ID = os.getenv("VK_GROUP_ID", "")
VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN", "")

SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))

MAX_CONCURRENT_GENERATIONS = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "2"))
MAX_TABLE_ROWS_IN_PROMPT = int(os.getenv("MAX_TABLE_ROWS_IN_PROMPT", "30"))
MAX_TABLE_TOKENS_IN_PROMPT = int(os.getenv("MAX_TABLE_TOKENS_IN_PROMPT", "1500"))
USE_USHAPED_CONTEXT = os.getenv("USE_USHAPED_CONTEXT", "true").lower() == "true"

# ★ ИЗМЕНЕНО: параметры индексации (настраиваются под железо)
INDEXING_BATCH_SIZE = int(os.getenv("INDEXING_BATCH_SIZE", "256"))       # 256 для CPU/VPS, 1024+ для GPU
INDEXING_FLUSH_THRESHOLD = int(os.getenv("INDEXING_FLUSH_THRESHOLD", "2000"))  # 2000 для 2GB, 50000 для 20GB
INDEXING_DEVICE = os.getenv("INDEXING_DEVICE", "auto")                   # auto | cpu | cuda

def get_device() -> str:
    """Определить устройство для вычислений."""
    if INDEXING_DEVICE != "auto":
        return INDEXING_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def ensure_dirs() -> None:
    for d in [DOCS_DIR, OUTPUT_DIR, INDEX_DIR, DOCSTORE_DIR, FIGURES_DIR, LOGS_DIR, METADATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
