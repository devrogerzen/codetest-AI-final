"""Base vectorial: ChromaDB, como servidor (docker-compose) o como carpeta local persistente.

Guarda en los metadatos de la colección qué modelo de embeddings generó los vectores. Si después se abre
con otro modelo, falla con un mensaje claro: comparar vectores de dos modelos distintos no tiene sentido
y el error, si no se controla, es silencioso (la búsqueda "funciona" pero trae cualquier cosa).
"""
import logging
import time

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import CHROMA_DIR, CHROMA_HOST, CHROMA_PORT, nombre_coleccion, nombre_embeddings

log = logging.getLogger("rag")

class EmbeddingsIncompatibles(RuntimeError):
    pass


def cliente_chroma(espera_maxima: float = 60) -> chromadb.ClientAPI:
    ajustes = Settings(anonymized_telemetry=False)
    if not CHROMA_HOST:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=ajustes)
    # En docker-compose el servidor de Chroma puede tardar unos segundos en aceptar conexiones
    limite = time.monotonic() + espera_maxima
    while True:
        try:
            return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=ajustes)
        except Exception as e:
            if time.monotonic() > limite:
                raise ConnectionError(f"Chroma no responde en {CHROMA_HOST}:{CHROMA_PORT}") from e
            log.info("Esperando a Chroma en %s:%s...", CHROMA_HOST, CHROMA_PORT)
            time.sleep(2)


def abrir_almacen(embeddings: Embeddings, cliente: chromadb.ClientAPI | None = None,
                  coleccion: str | None = None) -> Chroma:
    cliente = cliente or cliente_chroma()
    coleccion = coleccion or nombre_coleccion()
    modelo = nombre_embeddings(embeddings)
    if coleccion in {c.name for c in cliente.list_collections()}:
        guardado = (cliente.get_collection(coleccion).metadata or {}).get("embeddings")
        if guardado and guardado != modelo:
            raise EmbeddingsIncompatibles(
                f"La colección '{coleccion}' se indexó con {guardado} y ahora se abre con {modelo}. "
                "Reindexá con `python -m app.rag.ingesta --reiniciar` o usá otra CHROMA_COLLECTION.")
    # Distancia coseno: la estándar para comparar embeddings de texto (ignora el largo del vector)
    return Chroma(collection_name=coleccion, embedding_function=embeddings, client=cliente,
                  collection_metadata={"embeddings": modelo, "hnsw:space": "cosine"})


def cantidad_de_fragmentos(almacen: Chroma) -> int:
    return len(almacen.get(include=[])["ids"])
