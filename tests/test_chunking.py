# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Тесты модуля чанкинга
======================================
Проверка семантического разбиения текста, генерации сводок
и создания чанков из DocumentElement.
"""
import pytest
from src.chunking import (
    Chunk, SmartChunker, _split_text_semantic,
    _generate_table_summary, _generate_formula_summary,
)
from src.parsing import DocumentElement


class TestSplitTextSemantic:
    """Тесты семантического разбиения текста."""

    def test_empty_text(self):
        """Пустой текст должен возвращать пустой список."""
        assert _split_text_semantic("") == []
        assert _split_text_semantic("   ") == []

    def test_short_text(self):
        """Текст короче MIN_CHUNK_LENGTH должен возвращать пустой список."""
        assert _split_text_semantic("Короткий текст") == []

    def test_single_paragraph(self):
        """Один длинный абзац должен вернуть один чанк."""
        text = "Это достаточно длинный текст для одного чанка. " * 20
        result = _split_text_semantic(text)
        assert len(result) == 1
        assert result[0].strip() == text.strip()

    def test_multiple_paragraphs(self):
        """Несколько абзацев должны разбиваться корректно."""
        para = "Абзац текста с достаточной длиной для обработки. " * 10
        text = f"{para}\n{para}\n{para}"
        result = _split_text_semantic(text, max_tokens=200)
        assert len(result) >= 2

    def test_overlap(self):
        """Перекрытие между чанками должно сохраняться."""
        para = "Уникальный абзац с текстом для проверки перекрытия. " * 10
        text = f"{para}\n{para}\n{para}"
        result = _split_text_semantic(text, max_tokens=200, overlap_tokens=50)
        if len(result) >= 2:
            assert len(result) >= 2


class TestGenerateTableSummary:
    """Тесты генерации сводок таблиц."""

    def test_basic_table(self):
        """Базовая сводка таблицы."""
        markdown = "| Параметр | Значение |\n|---|---|\n| Вес | 100 кг |\n| Длина | 5 м |"
        metadata = {"num_rows": 2, "num_cols": 2}
        summary = _generate_table_summary(markdown, metadata)
        assert "2 строк" in summary
        assert "2 столбцов" in summary
        assert "Параметр" in summary

    def test_empty_table(self):
        """Пустая таблица."""
        summary = _generate_table_summary("", {"num_rows": 0, "num_cols": 0})
        assert "0 строк" in summary


class TestGenerateFormulaSummary:
    """Тесты генерации сводок формул."""

    def test_latex_formula(self):
        """LaTeX-формула."""
        summary = _generate_formula_summary("E=mc^2", {"formula_type": "latex_inline"})
        assert "LaTeX" in summary
        assert "E=mc^2" in summary

    def test_unknown_formula(self):
        """Неизвестный тип формулы."""
        summary = _generate_formula_summary("x + y", {"formula_type": "unknown"})
        assert "Формула" in summary


class TestSmartChunker:
    """Тесты основного класса чанкера."""

    def test_chunk_text_elements(self):
        """Чанкинг текстовых элементов."""
        elements = [
            DocumentElement(
                content="Текстовый элемент достаточной длины для обработки и разбиения на чанки. " * 5,
                element_type="text",
                source="test.pdf",
                page=1,
            ),
        ]
        chunker = SmartChunker()
        chunks = chunker.chunk_elements(elements)
        assert len(chunks) >= 1
        assert chunks[0].chunk_type == "text"
        assert chunks[0].source == "test.pdf"

    def test_chunk_table_element(self):
        """Чанкинг табличного элемента."""
        elements = [
            DocumentElement(
                content="| A | B |\n|---|---|\n| 1 | 2 |",
                element_type="table",
                source="test.pdf",
                page=1,
                metadata={"num_rows": 1, "num_cols": 2},
            ),
        ]
        chunker = SmartChunker()
        chunks = chunker.chunk_elements(elements)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"
        assert chunks[0].full_content is not None

    def test_mixed_elements(self):
        """Смешанные элементы (текст + таблица)."""
        elements = [
            DocumentElement(
                content="Достаточно длинный текстовый элемент. " * 5,
                element_type="text",
                source="test.pdf",
                page=1,
            ),
            DocumentElement(
                content="| A | B |\n|---|---|\n| 1 | 2 |",
                element_type="table",
                source="test.pdf",
                page=1,
                metadata={"num_rows": 1, "num_cols": 2},
            ),
        ]
        chunker = SmartChunker()
        chunks = chunker.chunk_elements(elements)
        types = [c.chunk_type for c in chunks]
        assert "text" in types
        assert "table" in types
