"""Buscador híbrido: BM25 (palabras exactas) + vectores (significado), fusionados con Reciprocal Rank Fusion.

- BM25 encuentra términos técnicos y nombres propios que los embeddings diluyen (`pagos_modo_cola`, "429").
- Los vectores encuentran paráfrasis que BM25 no ve ("contraseña" -> "secretos filtrados").
- EnsembleRetriever fusiona las dos listas con RRF; `id_key="chunk_id"` hace que un mismo fragmento
  encontrado por los dos buscadores cuente una sola vez (y suba en el ranking).
- Los dos índices salen de la misma colección de Chroma, así que nunca quedan desincronizados.
- Filtro por metadatos (`categoria`): en Chroma se aplica dentro de la búsqueda; en BM25, al armar el índice.
"""
import logging

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from app.config import PESO_BM25, TOP_K, crear_embeddings
from app.rag.almacen import abrir_almacen
from app.texto import tokenizar

log = logging.getLogger("rag")

MODOS = ("hibrido", "vectorial", "bm25")


class BaseVacia(RuntimeError):
    pass


class BuscadorHibrido:
    CANDIDATOS = 8   # cada buscador trae más candidatos de los que se devuelven: la fusión elige los mejores

    def __init__(self, almacen: Chroma, k: int = TOP_K, peso_bm25: float = PESO_BM25):
        self.almacen = almacen
        self.k = k
        self.peso_bm25 = peso_bm25
        datos = almacen.get(include=["documents", "metadatas"])
        if not datos["ids"]:
            raise BaseVacia("La base vectorial está vacía: corré la ingesta (python -m app.rag.ingesta).")
        self.fragmentos = [Document(id=i, page_content=texto, metadata=meta)
                           for i, texto, meta in zip(datos["ids"], datos["documents"], datos["metadatas"])]
        self.categorias = sorted({f.metadata["categoria"] for f in self.fragmentos})
        self._bm25: dict[str | None, BM25Retriever] = {}

    @classmethod
    def desde_config(cls, embeddings: Embeddings | None = None, **kwargs) -> "BuscadorHibrido":
        return cls(abrir_almacen(embeddings or crear_embeddings()), **kwargs)

    def _retriever_bm25(self, categoria: str | None) -> BM25Retriever:
        if categoria not in self._bm25:
            docs = [f for f in self.fragmentos if categoria is None or f.metadata["categoria"] == categoria]
            if not docs:
                raise ValueError(f"No hay documentos con la categoría {categoria!r} (hay: {', '.join(self.categorias)})")
            self._bm25[categoria] = BM25Retriever.from_documents(docs, k=self.CANDIDATOS, preprocess_func=tokenizar)
        return self._bm25[categoria]

    def _retriever_vectorial(self, categoria: str | None) -> BaseRetriever:
        busqueda: dict = {"k": self.CANDIDATOS}
        if categoria:
            busqueda["filter"] = {"categoria": {"$eq": categoria}}
        return self.almacen.as_retriever(search_kwargs=busqueda)

    def retriever(self, modo: str = "hibrido", categoria: str | None = None) -> BaseRetriever:
        if modo == "bm25":
            return self._retriever_bm25(categoria)
        if modo == "vectorial":
            return self._retriever_vectorial(categoria)
        if modo == "hibrido":
            return EnsembleRetriever(retrievers=[self._retriever_bm25(categoria), self._retriever_vectorial(categoria)],
                                     weights=[self.peso_bm25, 1 - self.peso_bm25], id_key="chunk_id")
        raise ValueError(f"Modo desconocido: {modo!r} (opciones: {', '.join(MODOS)})")

    async def buscar(self, consulta: str, k: int | None = None, categoria: str | None = None,
                     modo: str = "hibrido") -> list[Document]:
        """Los k fragmentos más relevantes. Es async de punta a punta: la búsqueda vectorial (embedding de la
        consulta + Chroma) corre fuera del event loop y en paralelo con BM25."""
        docs = (await self.retriever(modo, categoria).ainvoke(consulta))[: k or self.k]
        log.info("[%s%s] %s -> %s", modo, f"/{categoria}" if categoria else "", consulta,
                 [d.metadata["chunk_id"] for d in docs])
        return docs
