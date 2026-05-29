# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Тесты модуля генерации
======================================
Проверка формирования промптов и работы LLMClient
(мокирование сетевых запросов).
"""
import pytest
from unittest.mock import patch, MagicMock
from src.generation import build_prompt, LLMClient
from src.retrieval import RetrievalResult


class TestBuildPrompt:
    """Тесты формирования промпта."""

    def test_basic_prompt(self):
        """Базовый промпт с одним результатом."""
        results = [
            RetrievalResult(
                chunk_id="test_1",
                content="Текст контекста",
                source="doc.pdf",
                page=1,
                chunk_type="text",
            ),
        ]
        prompt = build_prompt("Какой-то вопрос", results)
        # Исправлено: проверка актуальных XML-тегов вместо устаревших "КОНТЕКСТ:"
        assert "<retrieved_context>" in prompt
        assert "<user_query>" in prompt
        assert "Какой-то вопрос" in prompt
        assert "doc.pdf" in prompt

    def test_multiple_results(self):
        """Промпт с несколькими результатами."""
        results = [
            RetrievalResult(
                chunk_id=f"test_{i}",
                content=f"Контекст {i}",
                source=f"doc_{i}.pdf",
                page=i,
                chunk_type="text",
            )
            for i in range(3)
        ]
        prompt = build_prompt("Вопрос", results)
        assert "Фрагмент 1" in prompt
        assert "Фрагмент 2" in prompt
        assert "Фрагмент 3" in prompt

    def test_table_in_context(self):
        """Промпт с таблицей в контексте."""
        results = [
            RetrievalResult(
                chunk_id="test_t",
                content="Сводка",
                full_content="| A | B |\n|---|---|\n| 1 | 2 |",
                source="doc.pdf",
                page=1,
                chunk_type="table",
            ),
        ]
        prompt = build_prompt("Запрос", results)
        assert "ТАБЛИЦА:" in prompt


class TestLLMClient:
    """Тесты LLMClient с мокированием сетевых запросов."""

    def test_unsupported_provider(self):
        """Неподдерживаемый провайдер."""
        client = LLMClient(provider="unknown")
        result = client.generate("Тест")
        assert "не поддерживается" in result

    def test_gigachat_no_credentials(self):
        """GigaChat без учётных данных."""
        with patch("src.generation.GIGACHAT_CREDENTIALS", ""):
            client = LLMClient(provider="gigachat")
            result = client.generate("Тест")
            assert "не заданы" in result

    def test_openai_no_key(self):
        """OpenAI без API-ключа."""
        with patch("src.generation.OPENAI_API_KEY", ""):
            client = LLMClient(provider="openai")
            result = client.generate("Тест")
            assert "не задан" in result
