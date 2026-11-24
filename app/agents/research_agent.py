"""Agente de investigación: consulta la base de conocimiento (RAG híbrido) y las reseñas de clientes.

Es un agente ReAct (create_react_agent) con dos herramientas. Recibe la instrucción del supervisor, no el
historial completo, y escribe únicamente en state["hallazgos"]. Cada hallazgo de documentación lleva el
documento_id exacto que devolvió el buscador: es lo que el informe tiene que citar y lo que el validador
controla.
"""
import asyncio
from datetime import date
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.agents.comun import log, observaciones, siguiente_paso
from app.config import RECURSION_LIMIT_ESPECIALISTA
from app.datos import resenas_entre
from app.rag.buscador import BuscadorHibrido
from app.state import Aporte, EstadoOrquestador, Hallazgo

PROMPT = """Sos el investigador de un equipo de análisis de Colibrí, una tienda online. Tu único trabajo es \
conseguir información con tus herramientas: no calculás métricas ni sacás conclusiones.
- buscar_documentacion: busca en la base de conocimiento (arquitectura, runbooks, postmortems, despliegue, \
seguridad, preguntas frecuentes). Escribí la consulta con las palabras que usaría el documento. Si sabés de \
qué tipo es, filtrá por categoria (postmortems, operaciones, arquitectura, devops, seguridad).
- buscar_resenas: para saber qué opinan los clientes en un período. Para comparar dos períodos, una llamada por período.
Si necesitás una fecha que no te dieron (por ejemplo, la del incidente), buscala primero en la documentación.
Al terminar, respondé en dos o tres líneas qué encontraste: documentos y cantidad de reseñas por período. \
No inventes datos que no vengan de las herramientas."""

Categoria = Literal["postmortems", "operaciones", "arquitectura", "devops", "seguridad"]


class BuscarDocumentacionInput(BaseModel):
    consulta: str = Field(min_length=3, description="Qué buscar, con palabras del dominio: 'incidente pagos octubre impacto clientes'")
    categoria: Categoria | None = Field(default=None, description="Opcional: restringe la búsqueda a una categoría")
    k: int = Field(default=4, ge=1, le=6, description="Cantidad de fragmentos a devolver")


class BuscarResenasInput(BaseModel):
    desde: date = Field(description="Primer día del período, formato AAAA-MM-DD")
    hasta: date = Field(description="Último día del período (inclusive), formato AAAA-MM-DD")
    tema: str | None = Field(default=None, description="Opcional: palabra que tiene que aparecer en la reseña, como 'pago'")

    @model_validator(mode="after")
    def periodo_valido(self) -> "BuscarResenasInput":
        if self.hasta < self.desde:
            raise ValueError("'hasta' no puede ser anterior a 'desde'")
        return self


def crear_herramientas(buscador: BuscadorHibrido) -> list:
    @tool(args_schema=BuscarDocumentacionInput)
    async def buscar_documentacion(consulta: str, categoria: str | None = None, k: int = 4) -> dict[str, Any]:
        """Busca en la base de conocimiento interna de Colibrí (arquitectura, guía de incidentes y runbooks,
        postmortems, despliegue, seguridad y preguntas frecuentes) con búsqueda híbrida: palabras exactas y
        significado. Devuelve documento_id, título, sección y contenido de los fragmentos más relevantes. Usala
        para saber qué pasó en un incidente, cuándo fue, a cuántos clientes afectó o qué dice un procedimiento."""
        docs = await buscador.buscar(consulta, k=k, categoria=categoria)
        if not docs:
            return {"error": f"No encontré nada para '{consulta}'. Probá con otras palabras o sin categoría."}
        return {"resultados": [{"documento_id": d.metadata["documento_id"], "titulo": d.metadata["titulo"],
                                "seccion": d.metadata["seccion"], "categoria": d.metadata["categoria"],
                                "contenido": d.page_content} for d in docs]}

    @tool(args_schema=BuscarResenasInput)
    async def buscar_resenas(desde: date, hasta: date, tema: str | None = None) -> dict[str, Any]:
        """Trae las reseñas de clientes de Colibrí publicadas entre dos fechas, inclusive: id, fecha, calificación
        (1 a 5 estrellas), canal y texto. Usala para saber qué opinan los clientes en un período. Para comparar
        dos momentos (por ejemplo antes y después de un incidente), llamala una vez por cada período."""
        resenas = await asyncio.to_thread(resenas_entre, desde, hasta, tema)
        return {"periodo": f"{desde.isoformat()}/{hasta.isoformat()}", "tema": tema, "cantidad": len(resenas),
                "resenas": resenas}

    return [buscar_documentacion, buscar_resenas]


def extraer_hallazgos(mensajes: list, ya_vistos: set[str]) -> list[Hallazgo]:
    """Convierte las respuestas de las herramientas en Hallazgos validados (sin repetir referencias).
    Si el buscador trae varios fragmentos de la misma sección, se juntan en un solo hallazgo."""
    hallazgos: dict[str, Hallazgo] = {}
    for herramienta, resultado in observaciones(mensajes):
        if not isinstance(resultado, dict) or "error" in resultado:
            continue
        try:
            if herramienta == "buscar_documentacion":
                nuevos = [Hallazgo(tipo="documento", referencia=r["documento_id"], contenido=r["contenido"])
                          for r in resultado["resultados"]]
            else:
                ids = [r["id"] for r in resultado["resenas"]]
                promedio = sum(r["calificacion"] for r in resultado["resenas"]) / len(ids) if ids else 0
                nuevos = [Hallazgo(tipo="resenas", referencia=f"resenas {resultado['periodo']}",
                                   contenido=f"{len(ids)} reseñas del período {resultado['periodo']}"
                                             f" (calificación media cruda {promedio:.2f})", ids_resenas=ids)]
        except (KeyError, ValidationError) as e:   # un resultado mal formado no entra al estado
            log.warning("Hallazgo descartado de %s: %s", herramienta, e)
            continue
        for h in nuevos:
            if h.referencia in ya_vistos:
                continue
            if h.referencia in hallazgos:
                previo = hallazgos[h.referencia]
                if h.contenido not in previo.contenido:
                    hallazgos[h.referencia] = previo.model_copy(update={"contenido": f"{previo.contenido}\n{h.contenido}"})
            else:
                hallazgos[h.referencia] = h
    return list(hallazgos.values())


def crear_investigador(modelo: BaseChatModel, buscador: BuscadorHibrido):
    agente = create_react_agent(modelo, crear_herramientas(buscador), prompt=PROMPT, name="investigador")

    async def investigador(state: EstadoOrquestador) -> dict:
        # Contexto acotado: el pedido y la instrucción del supervisor, no toda la conversación
        tarea = HumanMessage(f"Pedido del usuario: {state['pedido']}\nTu tarea: {state['instruccion']}")
        resultado = await agente.ainvoke({"messages": [tarea]}, {"recursion_limit": RECURSION_LIMIT_ESPECIALISTA})
        hallazgos = extraer_hallazgos(resultado["messages"], {h.referencia for h in state["hallazgos"]})
        resumen = resultado["messages"][-1].text or f"{len(hallazgos)} hallazgos nuevos."
        return {
            "hallazgos": hallazgos,
            "aportes": [Aporte(paso=siguiente_paso(state), agente="investigador",
                               resumen=f"{resumen} [{', '.join(h.referencia for h in hallazgos) or 'sin hallazgos nuevos'}]")],
            "messages": [AIMessage(resumen, name="investigador")],
        }

    return investigador
