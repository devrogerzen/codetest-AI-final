"""Evaluación del buscador con un golden set: preguntas con el documento que debería aparecer.

Métricas, a nivel documento_id (un documento con varios fragmentos cuenta una vez):
- Recall@k: de los documentos relevantes, qué parte apareció en los k primeros. Si es bajo, el modelo no ve
  la respuesta y alucina por falta de contexto.
- Precision@k: de los k primeros, qué parte es relevante. Si es baja, entra ruido al prompt.
- MRR: promedio de 1 / posición del primer relevante. Premia que lo bueno aparezca arriba.
"""
import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from app.config import GOLDEN_SET
from app.rag.buscador import BuscadorHibrido


def cargar_golden_set(ruta: Path = GOLDEN_SET) -> list[dict[str, Any]]:
    return json.loads(ruta.read_text(encoding="utf-8"))


def relevantes(item: dict[str, Any]) -> set[str]:
    esperado = item["documento_id_esperado"]
    return {esperado} if isinstance(esperado, str) else set(esperado)


def recall_at_k(recuperados: list[str], relevantes_: set[str], k: int) -> float:
    return len(relevantes_ & set(recuperados[:k])) / len(relevantes_)


def precision_at_k(recuperados: list[str], relevantes_: set[str], k: int) -> float:
    return len(relevantes_ & set(recuperados[:k])) / k


def reciprocal_rank(recuperados: list[str], relevantes_: set[str]) -> float:
    return next((1 / posicion for posicion, doc in enumerate(recuperados, 1) if doc in relevantes_), 0.0)


async def evaluar(buscador: BuscadorHibrido, golden: list[dict[str, Any]], modo: str, k: int,
                  concurrencia: int = 4) -> dict[str, Any]:
    """Corre todas las preguntas (de a `concurrencia` a la vez, para no chocar con el límite del proveedor)."""
    semaforo = asyncio.Semaphore(concurrencia)

    async def una(item: dict[str, Any]) -> dict[str, Any]:
        async with semaforo:
            docs = await buscador.buscar(item["pregunta"], k=k, modo=modo)
        recuperados = list(dict.fromkeys(d.metadata["documento_id"] for d in docs))
        rel = relevantes(item)
        return {"pregunta": item["pregunta"], "esperados": sorted(rel), "recuperados": recuperados,
                "recall": recall_at_k(recuperados, rel, k), "precision": precision_at_k(recuperados, rel, k),
                "rr": reciprocal_rank(recuperados, rel)}

    detalle = await asyncio.gather(*(una(item) for item in golden))
    promedio = lambda clave: round(sum(d[clave] for d in detalle) / len(detalle), 3)  # noqa: E731
    return {"modo": modo, "k": k, "recall": promedio("recall"), "precision": promedio("precision"),
            "mrr": promedio("rr"), "detalle": detalle}


class EmbeddingsConCache(Embeddings):
    """Envuelve un modelo de embeddings y recuerda cada texto ya calculado: en los experimentos (varios
    modos, pesos y tamaños de fragmento) cada pregunta y cada fragmento se paga una sola vez."""

    def __init__(self, base: Embeddings):
        self.base = base
        self.model = getattr(base, "model", type(base).__name__)
        self._cache: dict[tuple[str, str], list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        faltan = [t for t in dict.fromkeys(texts) if ("doc", t) not in self._cache]
        if faltan:
            for texto, vector in zip(faltan, self.base.embed_documents(faltan)):
                self._cache[("doc", texto)] = vector
        return [self._cache[("doc", t)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if ("query", text) not in self._cache:
            self._cache[("query", text)] = self.base.embed_query(text)
        return self._cache[("query", text)]
