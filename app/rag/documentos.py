"""Carga de la base de conocimiento (Markdown y FAQ en JSON), chunking por tokens y metadatos.

- Cada sección (##) de un Markdown y cada entrada del FAQ es un documento con un `documento_id` estable,
  por ejemplo "guia_incidentes.md#runbook-errores-5xx-en-pagos". Es lo que citan los agentes y lo que
  evalúa el golden set.
- Cada documento se corta en fragmentos de hasta CHUNK_TOKENS tokens (tiktoken), con solapamiento.
- Cada fragmento lleva un ID determinista (hash de documento, posición y texto): la ingesta es
  idempotente y solo vuelve a pagar el embedding de lo que cambió.
- Cada fragmento empieza con "Título del documento > Sección": así un fragmento suelto no pierde de qué
  habla, y tanto BM25 como los embeddings encuentran la sección por su nombre.
"""
import hashlib
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_TOKENS, CONOCIMIENTO, OVERLAP_TOKENS
from app.texto import slug

CATALOGO = "metadatos.json"   # categoría, etiquetas y fecha de cada archivo Markdown


def _secciones(texto: str) -> tuple[str, list[tuple[str, str]]]:
    """Título del documento (#) y sus secciones (##) como (nombre, cuerpo). Lo que va antes de la primera
    sección es la introducción; se descarta si es solo la nota de "documentación ficticia"."""
    titulo = texto.splitlines()[0].lstrip("# ").strip()
    bloques = re.split(r"^## ", texto, flags=re.MULTILINE)
    intro = "\n".join(linea for linea in bloques[0].splitlines()[1:] if not linea.startswith(">")).strip()
    secciones = [("Introducción", intro)] if len(intro) > 40 else []
    for bloque in bloques[1:]:
        nombre, _, cuerpo = bloque.partition("\n")
        secciones.append((nombre.strip(), cuerpo.strip()))
    return titulo, secciones


def cargar_documentos(directorio: Path = CONOCIMIENTO) -> list[Document]:
    """Un Document por sección de Markdown o por entrada del FAQ, con metadatos para filtrar y citar."""
    catalogo = json.loads((directorio / CATALOGO).read_text(encoding="utf-8"))
    documentos: list[Document] = []

    for archivo in sorted(directorio.glob("*.md")):
        info = catalogo.get(archivo.name, {})
        titulo, secciones = _secciones(archivo.read_text(encoding="utf-8"))
        for orden, (nombre, cuerpo) in enumerate(secciones, 1):
            documentos.append(Document(page_content=cuerpo, metadata={
                "documento_id": f"{archivo.name}#{slug(nombre)}",
                "fuente": archivo.name,
                "titulo": titulo,
                "seccion": nombre,
                "orden": orden,
                "categoria": info.get("categoria", "general"),
                "etiquetas": ",".join(info.get("etiquetas", [])),   # Chroma solo guarda valores simples
                "fecha": info.get("fecha", ""),
                "tipo": "markdown",
            }))

    for archivo in sorted(directorio.glob("*.json")):
        if archivo.name == CATALOGO:
            continue
        for orden, entrada in enumerate(json.loads(archivo.read_text(encoding="utf-8")), 1):
            documentos.append(Document(page_content=entrada["respuesta"], metadata={
                "documento_id": f"{archivo.name}#{entrada['id']}",
                "fuente": archivo.name,
                "titulo": "Preguntas frecuentes",
                "seccion": entrada["pregunta"],
                "orden": orden,
                "categoria": entrada.get("categoria", "general"),
                "etiquetas": ",".join(entrada.get("etiquetas", [])),
                "fecha": "",
                "tipo": "faq",
            }))

    if not documentos:
        raise FileNotFoundError(f"No hay documentos en {directorio}")
    return documentos


def fragmentar(documentos: list[Document], chunk_tokens: int = CHUNK_TOKENS,
               overlap_tokens: int = OVERLAP_TOKENS) -> list[Document]:
    """Corta cada documento por tokens: primero por párrafos, después por líneas y después por palabras."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=chunk_tokens, chunk_overlap=overlap_tokens)
    fragmentos: list[Document] = []
    for doc in documentos:
        documento_id = doc.metadata["documento_id"]
        encabezado = f"{doc.metadata['titulo']} > {doc.metadata['seccion']}"
        for i, texto in enumerate(splitter.split_text(doc.page_content)):
            contenido = f"{encabezado}\n\n{texto}"
            huella = hashlib.sha256(f"{documento_id}|{i}|{contenido}".encode()).hexdigest()[:16]
            fragmentos.append(Document(id=huella, page_content=contenido, metadata={
                **doc.metadata, "fragmento": i, "chunk_id": f"{documento_id}::{i}"}))
    return fragmentos


def categorias(directorio: Path = CONOCIMIENTO) -> list[str]:
    return sorted({d.metadata["categoria"] for d in cargar_documentos(directorio)})
