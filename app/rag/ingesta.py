"""Ingesta: carga data/conocimiento, fragmenta, genera los embeddings y los guarda en ChromaDB.

Es incremental e idempotente: compara los IDs deterministas de los fragmentos actuales con los que ya
están en la colección, sube solo los nuevos o modificados y borra los obsoletos. Correrla dos veces
seguidas no hace nada la segunda vez (y no gasta embeddings).

Uso:
    python -m app.rag.ingesta               # sincroniza la colección con la carpeta
    python -m app.rag.ingesta --reiniciar   # borra la colección y sube todo de cero
"""
import argparse
import asyncio
import json
import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import crear_embeddings, nombre_coleccion, nombre_embeddings, resumir_error
from app.rag.almacen import abrir_almacen, cliente_chroma
from app.rag.documentos import cargar_documentos, fragmentar

log = logging.getLogger("ingesta")


def sincronizar(almacen: Chroma, fragmentos: list[Document]) -> dict[str, int]:
    """Deja la colección igual a los fragmentos actuales, sin volver a subir lo que no cambió."""
    actuales = {f.id: f for f in fragmentos}
    existentes = set(almacen.get(include=[])["ids"])
    nuevos = [f for id_, f in actuales.items() if id_ not in existentes]
    obsoletos = sorted(existentes - actuales.keys())
    if obsoletos:
        almacen.delete(ids=obsoletos)
    if nuevos:
        almacen.add_documents(nuevos, ids=[f.id for f in nuevos])
    return {"fragmentos": len(fragmentos), "subidos": len(nuevos), "borrados": len(obsoletos),
            "sin_cambios": len(fragmentos) - len(nuevos)}


def ingestar(embeddings: Embeddings | None = None, reiniciar: bool = False,
             cliente: chromadb.ClientAPI | None = None, fragmentos: list[Document] | None = None) -> dict:
    embeddings = embeddings or crear_embeddings()
    cliente = cliente or cliente_chroma()
    coleccion = nombre_coleccion()
    if reiniciar and coleccion in {c.name for c in cliente.list_collections()}:
        cliente.delete_collection(coleccion)
        log.info("Colección '%s' borrada: se sube todo de cero", coleccion)
    almacen = abrir_almacen(embeddings, cliente, coleccion)
    resumen = sincronizar(almacen, fragmentos if fragmentos is not None else fragmentar(cargar_documentos()))
    resumen |= {"coleccion": coleccion, "embeddings": nombre_embeddings(embeddings)}
    log.info("Ingesta en '%s': %d subidos, %d borrados, %d sin cambios", coleccion,
             resumen["subidos"], resumen["borrados"], resumen["sin_cambios"])
    return resumen


async def aingestar(**kwargs) -> dict:
    """Versión para código async: Chroma y el cliente de embeddings por lotes son síncronos, así que la
    ingesta corre en un hilo aparte y no bloquea el event loop."""
    return await asyncio.to_thread(ingestar, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for ruido in ("httpx", "httpx2", "chromadb"):
        logging.getLogger(ruido).setLevel(logging.WARNING)
    argumentos = argparse.ArgumentParser(description="Ingesta incremental de la base de conocimiento en ChromaDB")
    argumentos.add_argument("--reiniciar", action="store_true", help="borra la colección y sube todo")
    try:
        print(json.dumps(ingestar(reiniciar=argumentos.parse_args().reiniciar), ensure_ascii=False))
    except Exception as e:   # key faltante, Chroma caído, modelo de embeddings distinto...
        raise SystemExit(f"La ingesta falló: {resumir_error(e)}")
