# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Тесты модуля утилит
===================================
Проверка функций очистки текста, хэширования,
оценки токенов, детекции моделей.
"""
import pytest
from src.utils import (
    clean_text, text_hash, estimate_tokens, truncate_to_tokens,
    generate_unique_id, detect_model_in_text, detect_model_from_filename,
    find_figure_caption, extract_headings,
)


class TestCleanText:
    """Тесты очистки текста."""

    def test_empty_input(self):
        assert clean_text("") == ""

    def test_multiple_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_leading_trailing_spaces(self):
        text = "  строка 1\nстрока 2  "
        result = clean_text(text)
        assert result.startswith("строка 1")

    def test_control_characters(self):
        text = "текст\x00с\x01контрольными\x08символами"
        result = clean_text(text)
        assert "\x00" not in result
        assert "контрольными" in result


class TestTextHash:
    """Тесты хэширования."""

    def test_deterministic(self):
        """Один и тот же текст -> один и тот же хэш."""
        assert text_hash("тест") == text_hash("тест")

    def test_different_texts(self):
        """Разные тексты -> разные хэши."""
        assert text_hash("текст1") != text_hash("текст2")

    def test_length(self):
        """Хэш должен быть 16 символов."""
        assert len(text_hash("тест")) == 16


class TestEstimateTokens:
    """Тесты оценки токенов."""

    def test_empty_text(self):
        assert estimate_tokens("") == 0

    def test_non_empty(self):
        assert estimate_tokens("текст") >= 1

    def test_longer_text_more_tokens(self):
        assert estimate_tokens("длинный текст") > estimate_tokens("короткий")


class TestTruncateToTokens:
    """Тесты обрезки текста."""

    def test_short_text(self):
        assert truncate_to_tokens("тест", 100) == "тест"

    def test_empty(self):
        assert truncate_to_tokens("", 100) == ""


class TestGenerateUniqueId:
    """Тесты генерации ID."""

    def test_unique(self):
        id1 = generate_unique_id()
        id2 = generate_unique_id()
        assert id1 != id2

    def test_with_prefix(self):
        uid = generate_unique_id(prefix="txt_")
        assert uid.startswith("txt_")


class TestDetectModel:
    """Тесты детекции моделей."""

    def test_model_in_text(self):
        text = "Паспорт устройства АИИС-001"
        assert detect_model_in_text(text) == "АИИС-001"

    def test_no_model(self):
        assert detect_model_in_text("Обычный текст") == "unknown"

    def test_model_from_filename(self):
        assert "АИИС" in detect_model_from_filename("ПАСПОРТ_АИИС-001.pdf") or \
               detect_model_from_filename("ПАСПОРТ_АИИС-001.pdf") != "unknown"


class TestFindFigureCaption:
    """Тесты поиска подписей к рисункам."""

    def test_ris_caption(self):
        text = "Какой-то текст. Рис. 1 Схема подключения. Ещё текст."
        caption = find_figure_caption(text)
        assert "Схема подключения" in caption

    def test_no_caption(self):
        assert find_figure_caption("Обычный текст без рисунков") == ""


class TestExtractHeadings:
    """Тесты извлечения заголовков."""

    def test_numbered_heading(self):
        text = "1.2 Общие сведения\nОбычный текст"
        headings = extract_headings(text)
        assert "1.2 Общие сведения" in headings

    def test_uppercase_heading(self):
        text = "ВВЕДЕНИЕ\nОбычный текст"
        headings = extract_headings(text)
        assert "ВВЕДЕНИЕ" in headings

    def test_no_headings(self):
        assert extract_headings("Обычный текст без заголовков") == []
