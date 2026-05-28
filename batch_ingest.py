# -*- coding: utf-8 -*-
import time, logging, concurrent.futures, multiprocessing
from pathlib import Path
from typing import List, Optional, Set, Tuple

# ★ ПОДАВЛЕНИЕ СПАМА ОТ pymorphy2
logging.getLogger("pymorphy2").setLevel(logging.WARNING)
logging.getLogger("pymorphy2.opencorpora_dict").setLevel(logging.WARNING)

from config import DOCS_DIR, OUTPUT_DIR, ensure_dirs
from src.parsing import parse_pdf, parse_docx
from src.chunking import SmartChunker
from src.indexing import IndexBuilder

logger = logging.getLogger("koib.ingest")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}

def _process_file_task(file_path: Path) -> Tuple[Path, List, bool]:
    """
    Задача для пула потоков: парсинг и чанкинг одного файла.
    Вызывается в отдельном потоке, чтобы загрузить все ядра CPU.
    """
    ext = file_path.suffix.lower()
    try:
        if ext == ".pdf":
            elements = parse_pdf(file_path)
        elif ext in (".docx", ".doc"):
            elements = parse_docx(file_path)
        else:
            return file_path, [], False
            
        if not elements:
            return file_path, [], False
            
        chunker = SmartChunker()
        chunks = chunker.chunk_elements(elements)
        return file_path, chunks, True
    except Exception as e:
        logger.error(f"Ошибка обработки {file_path.name}: {e}")
        return file_path, [], False

class BatchIngester:
    def __init__(self, docs_dir: Optional[Path] = None, output_dir: Optional[Path] = None, incremental: bool = True):
        self.docs_dir = docs_dir or DOCS_DIR
        self.output_dir = output_dir or OUTPUT_DIR
        self.incremental = incremental
        self.index_builder = IndexBuilder(self.output_dir / "index")
        self._processed_files: Set[str] = set()
        if incremental: self._load_processed_files()

    def _load_processed_files(self):
        p = self.output_dir / "metadata" / "ingest_manifest.txt"
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f: 
                self._processed_files = set(l.strip() for l in f if l.strip())

    def _save_processed_files(self):
        p = self.output_dir / "metadata" / "ingest_manifest.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            for fn in sorted(self._processed_files): 
                f.write(f"{fn}\n")

    def _discover_files(self) -> List[Path]:
        if not self.docs_dir.exists(): return []
        
        all_files = set()
        for ext in SUPPORTED_EXTENSIONS:
            all_files.update(self.docs_dir.glob(f"**/*{ext}"))
            all_files.update(self.docs_dir.glob(f"**/*{ext.upper()}"))
            
        # ★ ИСПРАВЛЕНИЕ ДУБЛИРОВАНИЯ В WINDOWS:
        # Файловая система Windows нечувствительна к регистру.
        # Используем resolve() для приведения к каноническому пути.
        resolved_files = {f.resolve(): f for f in all_files}
        unique_files = list(resolved_files.values())
        
        if not self.incremental: return unique_files
        return [f for f in unique_files if f.name not in self._processed_files]

    def process_all(self) -> None:
        ensure_dirs()
        t0 = time.time()
        files = self._discover_files()
        if not files: 
            print("  Файлы для индексации не найдены.")
            return
        
        print(f"  Обнаружено файлов для индексации: {len(files)}")
        success_count = error_count = 0
        
        # ★ УСКОРЕНИЕ: Параллельная обработка во всех ядрах CPU
        # PyMuPDF и Tesseract OCR отлично работают в потоках (отпускают GIL)
        cpu_count = multiprocessing.cpu_count() or 1
        max_workers = min(32, cpu_count + 4)
        print(f"  Запуск параллельного парсинга в {max_workers} потоков (CPU: {cpu_count})...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(_process_file_task, fp): fp for fp in files}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file), 1):
                file_path = future_to_file[future]
                print(f"  [{i}/{len(files)}] {file_path.name}...", end=" ", flush=True)
                
                try:
                    fp, chunks, success = future.result()
                    if success and chunks:
                        print(f"[{len(chunks)} чанков] ", end="", flush=True)
                        # Индексация (SQLite/FAISS) в основном потоке для безопасности БД
                        self.index_builder.add_chunks(chunks)
                        success_count += 1
                        self._processed_files.add(fp.name)
                        print("OK")
                    else:
                        error_count += 1
                        print("ОШИБКА (пусто)")
                except Exception as e:
                    logger.error(f"Критическая ошибка {file_path.name}: {e}")
                    error_count += 1
                    print("КРИТИЧЕСКАЯ ОШИБКА")
                    
                # Периодически сохраняем манифест
                if i % 10 == 0:
                    self._save_processed_files()

        self._save_processed_files()
        self.index_builder.save()
        print(f"\nРезультат: {success_count} успешно, {error_count} ошибок")
        print(f"  Индексы успешно развернуты на диске. Время сборки: {time.time() - t0:.1f}с")
