# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Тесты модуля поиска
===================================
Проверка определения интента запроса и форматирования
результатов поиска.
"""
import pytest
from src.retrieval import (
    RetrievalResult, _detect_query_intent,
    ResponseCache,
)


class TestDetectQueryIntent:
    """Тесты определения интента запроса."""

    def test_text_intent(self):
        """Обычный запрос — текстовый интент."""
        intent = _detect_query_intent("Как работает система?")
        assert intent["text"] == 1.0
        assert intent["table"] == 0.0

    def test_table_intent(self):
        """Запрос про таблицу — табличный интент."""
        intent = _detect_query_intent("Покажи таблицу параметров")
        assert intent["table"] > 0.0
        assert intent["text"] < 1.0

    def test_formula_intent(self):
        """Запрос про формулу — формульный интент."""
        intent = _detect_query_intent("Какая формула расчёта коэффициента?")
        assert intent["formula"] > 0.0

    def test_figure_intent(self):
        """Запрос про схему — интент рисунка."""
        intent = _detect_query_intent("Покажи схему подключения")
        assert intent["figure"] > 0.0

    def test_mixed_intent(self):
        """Смешанный запрос."""
        intent = _detect_query_intent("Таблица значений и формула расчёта")
        assert intent["table"] > 0.0
        assert intent["formula"] > 0.0


class TestRetrievalResult:
    """Тесты форматирования результатов поиска."""

    def test_text_context_string(self):
        """Форматирование текстового результата."""
        r = RetrievalResult(
            chunk_id="test_1",
            content="Текстовый фрагмент",
            source="doc.pdf",
            page=5,
            chunk_type="text",
        )
        ctx = r.to_context_string()
        assert "doc.pdf" in ctx
        assert "стр. 5" in ctx
        assert "Текстовый фрагмент" in ctx

    def test_table_context_string(self):
        """Форматирование табличного результата."""
        r = RetrievalResult(
            chunk_id="test_2",
            content="Сводка таблицы",
            full_content="| A | B |\n|---|---|\n| 1 | 2 |",
            source="doc.pdf",
            page=3,
            chunk_type="table",
        )
        ctx = r.to_context_string()
        assert "ТАБЛИЦА:" in ctx
        assert "| A | B |" in ctx

    def test_formula_context_string(self):
        """Форматирование формульного результата."""
        r = RetrievalResult(
            chunk_id="test_3",
            content="Формула",
            full_content="E = mc^2",
            source="doc.pdf",
            page=10,
            chunk_type="formula",
        )
        ctx = r.to_context_string()
        assert "ФОРМУЛА:" in ctx
        assert "E = mc^2" in ctx


class TestResponseCache:
    """Тесты SQLite-кэша ответов HyDE."""

    def test_set_and_get(self, tmp_path):
        """Сохранение и получение значения из кэша."""
        cache = ResponseCache(path=tmp_path / "test_cache.db")
        cache.set("тестовый запрос", "гипотетический ответ")
        result = cache.get("тестовый запрос")
        assert result == "гипотетический ответ"

    def test_cache_miss(self, tmp_path):
        """Промах кэша."""
        cache = ResponseCache(path=tmp_path / "test_cache.db")
        result = cache.get("несуществующий запрос")
        assert result is None

    def test_cache_case_insensitive(self, tmp_path):
        """Кэш нечувствителен к регистру и пробелам."""
        cache = ResponseCache(path=tmp_path / "test_cache.db")
        cache.set("Тестовый Запрос", "ответ")
        result = cache.get("тестовый запрос")
        assert result == "ответ"
