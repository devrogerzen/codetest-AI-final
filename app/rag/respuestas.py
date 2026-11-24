"""Respuestas directas sobre la base de conocimiento (POST /ask y POST /ask/stream).

Es el RAG "corto" del sistema: para una pregunta puntual no hace falta el equipo de agentes. La cadena es
LCEL de punta a punta:

    pregunta -> RunnableParallel(fragmentos = buscador híbrido, pregunta = passthrough)
             -> assign(contexto = fragmentos con su chunk_id entre corchetes)
             -> prompt "filtro de veracidad" -> LLM con salida estructurada (RespuestaLLM)

Tres defensas contra la alucinación (las de la pre-entrega 3):
1. el prompt obliga a responder solo con el contexto y a decir "No lo sé." si no alcanza;
2. cada fuente que cita el modelo se verifica contra lo que se recuperó de verdad: las inventadas se descartan;
3. si el modelo dice que encontró la respuesta pero no cita ninguna fuente válida, se responde "No lo sé.".
"""
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

from app.rag.buscador import BuscadorHibrido

log = logging.getLogger("rag")

NO_LO_SE = "No lo sé."

SISTEMA = """Sos el asistente técnico de Colibrí y funcionás como un filtro de veracidad:
- Respondé SOLO con información del CONTEXTO, aunque sepas la respuesta por otro lado.
- Si el contexto no alcanza, respondé exactamente "{no_lo_se}" y marcá que no la encontraste.
- Citá los chunk_id (los que están entre corchetes) de los fragmentos que usaste."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SISTEMA),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
]).partial(no_lo_se=NO_LO_SE)

PROMPT_STREAM = ChatPromptTemplate.from_messages([
    ("system", SISTEMA + "\nEscribí la respuesta en texto plano y al final la lista de chunk_id citados entre corchetes."),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
]).partial(no_lo_se=NO_LO_SE)


class RespuestaLLM(BaseModel):
    """Lo que devuelve el modelo (salida estructurada)."""
    respuesta: str = Field(description="La respuesta, o exactamente 'No lo sé.'")
    encontrada_en_contexto: bool = Field(description="true solo si el contexto alcanzó para responder")
    fuentes: list[str] = Field(default_factory=list, description="chunk_id de los fragmentos usados")


class Fuente(BaseModel):
    chunk_id: str
    documento_id: str
    titulo: str
    seccion: str
    categoria: str


class Pregunta(BaseModel):
    pregunta: str = Field(min_length=5, max_length=500, examples=["¿Qué flag hay que activar si falla el proveedor de pagos?"])
    categoria: str | None = Field(default=None, description="Opcional: filtra por categoría (operaciones, seguridad...)")
    k: int = Field(default=4, ge=1, le=8, description="Fragmentos a recuperar")


class RespuestaRAG(BaseModel):
    pregunta: str
    respuesta: str
    encontrada: bool
    fuentes: list[Fuente]
    recuperados: list[str] = Field(description="chunk_id de todo lo que se le pasó al modelo")
    descartadas: list[str] = Field(default_factory=list, description="Fuentes que citó el modelo y no se habían recuperado")


def fuente(doc: Document) -> Fuente:
    m = doc.metadata
    return Fuente(chunk_id=m["chunk_id"], documento_id=m["documento_id"], titulo=m["titulo"],
                  seccion=m["seccion"], categoria=m["categoria"])


def formatear_contexto(docs: list[Document]) -> str:
    if not docs:
        return "(no se recuperó ningún fragmento)"
    return "\n\n---\n\n".join(f"[{d.metadata['chunk_id']}]\n{d.page_content}" for d in docs)


def verificar(pregunta: str, docs: list[Document], salida: RespuestaLLM) -> RespuestaRAG:
    """Defensas 2 y 3: nada de fuentes inventadas y nada de 'encontré' sin una fuente real."""
    recuperados = {d.metadata["chunk_id"]: d for d in docs}
    citadas = list(dict.fromkeys(salida.fuentes))
    validas = [c for c in citadas if c in recuperados]
    descartadas = [c for c in citadas if c not in recuperados]
    if descartadas:
        log.warning("El LLM citó fuentes que no se recuperaron (se descartan): %s", descartadas)
    encontrada = salida.encontrada_en_contexto and bool(validas)
    return RespuestaRAG(pregunta=pregunta, respuesta=salida.respuesta if encontrada else NO_LO_SE,
                        encontrada=encontrada, fuentes=[fuente(recuperados[c]) for c in validas] if encontrada else [],
                        recuperados=list(recuperados), descartadas=descartadas)


def construir_cadena(buscador: BuscadorHibrido, llm: BaseChatModel) -> Runnable:
    """La cadena LCEL. Recibe {"pregunta", "categoria", "k"} y devuelve una RespuestaRAG validada."""
    async def recuperar(entrada: dict[str, Any]) -> list[Document]:
        return await buscador.buscar(entrada["pregunta"], k=entrada.get("k"), categoria=entrada.get("categoria"))

    generar = PROMPT | llm.with_structured_output(RespuestaLLM).with_retry(stop_after_attempt=2)
    return (
        RunnableParallel(fragmentos=RunnableLambda(recuperar), pregunta=lambda e: e["pregunta"])
        | RunnablePassthrough.assign(contexto=lambda x: formatear_contexto(x["fragmentos"]))
        | RunnablePassthrough.assign(salida=generar)
        | RunnableLambda(lambda x: verificar(x["pregunta"], x["fragmentos"], x["salida"]))
    ).with_config(run_name="rag_respuesta")


async def responder_en_stream(buscador: BuscadorHibrido, llm: BaseChatModel, pregunta: Pregunta) -> AsyncIterator[str]:
    """Server-Sent Events: primero las fuentes recuperadas, después cada pedazo del texto apenas lo genera
    el modelo (baja el tiempo hasta la primera palabra) y al final qué fuentes citó de verdad."""
    docs = await buscador.buscar(pregunta.pregunta, k=pregunta.k, categoria=pregunta.categoria)
    yield _evento("fuentes", [fuente(d).model_dump() for d in docs])
    texto = ""
    cadena = (PROMPT_STREAM | llm | StrOutputParser()).with_config(run_name="generacion_stream")
    async for pedazo in cadena.astream({"contexto": formatear_contexto(docs), "pregunta": pregunta.pregunta}):
        texto += pedazo
        yield _evento("token", pedazo)
    recuperados = {d.metadata["chunk_id"] for d in docs}
    citadas = [c for c in recuperados if f"[{c}]" in texto or c in texto]
    yield _evento("fin", {"encontrada": NO_LO_SE not in texto and bool(citadas), "fuentes_citadas": sorted(citadas)})


def _evento(nombre: str, datos: Any) -> str:
    return f"event: {nombre}\ndata: {json.dumps(datos, ensure_ascii=False)}\n\n"
