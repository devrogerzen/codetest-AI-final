"""Evalúa el buscador con el golden set (data/golden_set.json): Recall@k, Precision@k y MRR.

Compara los tres modos (BM25, vectorial e híbrido), varios pesos de la fusión RRF y dos tamaños de fragmento.
Cada combinación se indexa en una base temporal (no toca la colección del sistema) y los embeddings se
calculan una sola vez por texto (EmbeddingsConCache), así el experimento completo cuesta unas pocas llamadas.

Uso:
    python scripts/evaluar_rag.py                         # embeddings del .env (Gemini en la corrida del README)
    python scripts/evaluar_rag.py --chunks 400 120 --pesos 0.3 0.5 0.7
"""
import argparse
import asyncio
import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

from app.config import CHUNK_TOKENS, PESO_BM25, TOP_K, crear_embeddings, nombre_embeddings  # noqa: E402
from app.rag.almacen import abrir_almacen  # noqa: E402
from app.rag.buscador import BuscadorHibrido  # noqa: E402
from app.rag.documentos import cargar_documentos, fragmentar  # noqa: E402
from app.rag.evaluacion import EmbeddingsConCache, cargar_golden_set, evaluar  # noqa: E402
from app.rag.ingesta import ingestar  # noqa: E402

RESULTADOS = Path(__file__).resolve().parent.parent / "resultados"


async def main(k: int, chunks: list[int], pesos: list[float], salida: str) -> None:
    golden = cargar_golden_set()
    embeddings = EmbeddingsConCache(crear_embeddings())
    modelo = nombre_embeddings(embeddings.base)
    print(f"Golden set: {len(golden)} preguntas · embeddings: {modelo} · k = {k}\n")
    filas, detalle_produccion = [], None
    inicio = time.perf_counter()
    for chunk in chunks:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as carpeta:
            cliente = chromadb.PersistentClient(path=carpeta, settings=Settings(anonymized_telemetry=False))
            fragmentos = fragmentar(cargar_documentos(), chunk_tokens=chunk, overlap_tokens=chunk // 10)
            ingestar(embeddings=embeddings, cliente=cliente, fragmentos=fragmentos)
            almacen = abrir_almacen(embeddings, cliente)
            combinaciones = [("bm25", None), ("vectorial", None)] + [("hibrido", p) for p in pesos]
            for modo, peso in combinaciones:
                buscador = BuscadorHibrido(almacen, k=k, peso_bm25=peso if peso is not None else PESO_BM25)
                r = await evaluar(buscador, golden, modo, k)
                filas.append({"chunk_tokens": chunk, "fragmentos": len(fragmentos), "modo": modo, "peso_bm25": peso,
                              "recall": r["recall"], "precision": r["precision"], "mrr": r["mrr"]})
                if chunk == CHUNK_TOKENS and modo == "hibrido" and peso == PESO_BM25:
                    detalle_produccion = r["detalle"]
            del almacen, cliente

    print(f"{'chunk':>6} {'frag.':>6} {'modo':<10} {'peso BM25':>9} {'Recall@k':>9} {'Precision@k':>12} {'MRR':>6}")
    for f in filas:
        peso = f"{f['peso_bm25']:.1f}" if f["peso_bm25"] is not None else "-"
        print(f"{f['chunk_tokens']:>6} {f['fragmentos']:>6} {f['modo']:<10} {peso:>9} {f['recall']:>9.3f} "
              f"{f['precision']:>12.3f} {f['mrr']:>6.3f}")
    if detalle_produccion:
        fallos = [d for d in detalle_produccion if d["recall"] < 1]
        print(f"\nConfiguración del sistema (chunk {CHUNK_TOKENS}, híbrido {PESO_BM25}): "
              f"{len(golden) - len(fallos)}/{len(golden)} preguntas con todos sus documentos en el top {k}")
        for d in fallos:
            print(f"  - {d['pregunta']}\n    esperaba {d['esperados']}, trajo {d['recuperados']}")

    RESULTADOS.mkdir(exist_ok=True)
    destino = RESULTADOS / salida
    destino.write_text(json.dumps({"fecha": datetime.now().isoformat(timespec="seconds"), "embeddings": modelo, "k": k,
                                   "preguntas": len(golden), "duracion_s": round(time.perf_counter() - inicio, 1),
                                   "resultados": filas, "detalle_configuracion_del_sistema": detalle_produccion},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado en resultados/{salida}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)
    argumentos = argparse.ArgumentParser(description="Evaluación del buscador con el golden set")
    argumentos.add_argument("--k", type=int, default=TOP_K)
    argumentos.add_argument("--chunks", type=int, nargs="+", default=[CHUNK_TOKENS, 120])
    argumentos.add_argument("--pesos", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    argumentos.add_argument("--salida", default="evaluacion_rag.json")
    a = argumentos.parse_args()
    asyncio.run(main(a.k, a.chunks, a.pesos, a.salida))
