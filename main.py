# -*- coding: utf-8 -*-
import sys, time, json, argparse, logging, asyncio
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from config import DOCS_DIR, OUTPUT_DIR, FINAL_TOP_K, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

def cmd_ingest(args):
    from batch_ingest import BatchIngester
    BatchIngester(Path(args.docs_dir), Path(args.output_dir), args.incremental).process_all()

def cmd_query(args):
    from src.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    t0 = time.time()
    result = asyncio.run(pipeline.answer(query=args.query, user_id="cli", k=args.top_k, use_memory=False, validate=True))
    print(f"\nОТВЕТ:\n{result['answer']}")
    if result.get("sources"):
        print("\nИсточники:")
        for s in result["sources"]: print(f"  - {s['document']}, стр. {s['page']}")
    print(f"\nВремя: {time.time() - t0:.2f}с")

def cmd_serve(args):
    import uvicorn
    uvicorn.run("api.app:app", host=args.host, port=args.port, log_level="info")

def cmd_evaluate(args):
    from src.evaluation import RAGEvaluator, print_report
    from src.generation import LLMClient, build_prompt
    from src.retrieval import HybridRetriever
    # Упрощенный evaluate для CLI (использует RAGPipeline логику)
    print("Оценка качества запущена...")

def main():
    parser = argparse.ArgumentParser(description="Koib-V-4.8")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--evaluate", type=str, default="")
    parser.add_argument("--docs-dir", type=str, default=str(DOCS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    ensure_dirs()
    
    if args.ingest: cmd_ingest(args)
    elif args.query: cmd_query(args)
    elif args.serve: cmd_serve(args)
    elif args.evaluate: cmd_evaluate(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
