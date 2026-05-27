# KOIB-V-4.6 — Оптимизированная RAG-система
> Экстремально оптимизированная Retrieval-Augmented Generation система для технической документации.
> Разработана для работы на слабом VPS: **1 vCPU / 2 ГБ ОЗУ**.

## Обзор
KOIB-V-4.6 — это система вопросов и ответов (QA) по технической документации, построенная на гибридном поиске (векторный + BM25) с переранжированием и LLM-генерацией ответов. Система готова к интеграции с VK Callback API для работы в качестве чат-бота во время реальных выборов.

### Ключевые особенности
- **Лёгкие модели**: `intfloat/multilingual-e5-small` (33 МБ) с динамической квантизацией PyTorch (int8)
- **SQLite вместо JSON**: DocStore и кэш HyDE на SQLite для минимального потребления RAM
- **SQLite FTS5 вместо rank_bm25**: Sparse-поиск работает напрямую с диска (zero-RAM overhead)
- **Только Tesseract OCR**: нулевое потребление RAM в Python (в отличие от EasyOCR)
- **Эвристические сводки**: таблицы и формулы описываются без LLM (мгновенная обработка)
- **Жёсткие таймауты и лимиты**: Semaphore для защиты от OOM-спайков при генерации
- **FastAPI сервер**: полностью асинхронный эндпоинт для VK Callback API с Rate Limiting

## Структура проекта
```text
koib-v4.6/
├── config.py               # Централизованная конфигурация
├── main.py                 # CLI-точка входа (--ingest, --query, --serve, --evaluate)
├── batch_ingest.py         # Пакетная индексация документов
├── requirements.txt        # Зависимости Python
├── .env.example            # Пример конфигурации окружения
├── README.md               # Этот файл
│
├── src/                    # Основной код системы
│   ├── utils.py            # Утилиты (хэши, очистка, токены, детекция моделей)
│   ├── parsing.py          # Парсинг PDF (PyMuPDF) и DOCX
│   ├── chunking.py         # Умное разбиение на чанки
│   ├── indexing.py         # FAISS + SQLite FTS5 + SQLite DocStore
│   ├── retrieval.py        # Гибридный поиск (vector + FTS5 + reranker + HyDE)
│   ├── generation.py       # LLM-клиенты (GigaChat, OpenAI, Ollama) + Semaphore
│   ├── validation.py       # Валидация ответов (неуверенность, источники, семантика)
│   ├── quarantine.py       # Карантин сомнительных чанков (SQLite)
│   ├── safety.py           # Фильтрация опасного контента
│   └── evaluation.py       # Оценка качества RAG (LLM-as-Judge, 4 метрики)
│
├── api/                    # FastAPI-слой
│   ├── app.py              # FastAPI приложение
│   ├── routes/
│   │   ├── health.py       # Health check эндпоинт
│   │   └── vk_callback.py  # VK Callback API обработчик (+ Rate Limiting)
│   └── middleware/
│       └── logging.py      # Логирование HTTP-запросов
│
└── tests/                  # Unit-тесты
