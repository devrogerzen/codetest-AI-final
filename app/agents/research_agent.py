"""Agente de investigación: consulta la documentación de Colibrí y las reseñas de clientes.

Es un agente ReAct (create_react_agent) con solo dos herramientas. Recibe la instrucción del
supervisor, no el historial completo, y escribe únicamente en state["hallazgos"].
"""
import asyncio
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.agents.comun import log, observaciones, siguiente_paso
from app.config import RECURSION_LIMIT_ESPECIALISTA
from app.datos import buscar_secciones, resenas_entre
from app.state import Aporte, EstadoOrquestador, Hallazgo

PROMPT = """Sos el investigador de un equipo de análisis de Colibrí, una tienda online. Tu único trabajo es \
conseguir información con tus herramientas: no calculás métricas ni sacás conclusiones.
- buscar_documentacion: para el contexto (qué pasó en un incidente, cómo funciona un servicio).
- buscar_resenas: para saber qué opinan los clientes en un período. Para comparar dos períodos, una llamada por período.
Si necesitás una fecha que no te dieron (por ejemplo, la del incidente), buscala primero en la documentación.
Al terminar, respondé en dos o tres líneas qué encontraste: documentos y cantidad de reseñas por período. \
No inventes datos que no vengan de las herramientas."""


class BuscarDocumentacionInput(BaseModel):
    consulta: str = Field(min_length=3, description="Palabras clave, por ejemplo 'postmortem incidente pagos octubre'")
    k: int = Field(default=3, ge=1, le=5, description="Cantidad de secciones a devolver")


class BuscarResenasInput(BaseModel):
    desde: date = Field(description="Primer día del período, formato AAAA-MM-DD")
    hasta: date = Field(description="Último día del período (inclusive), formato AAAA-MM-DD")
    tema: str | None = Field(default=None, description="Opcional: palabra que tiene que aparecer en la reseña, como 'pago'")

    @model_validator(mode="after")
    def periodo_valido(self) -> "BuscarResenasInput":
        if self.hasta < self.desde:
            raise ValueError("'hasta' no puede ser anterior a 'desde'")
        return self


@tool(args_schema=BuscarDocumentacionInput)
async def buscar_documentacion(consulta: str, k: int = 3) -> dict[str, Any]:
    """Busca en la documentación interna de Colibrí (arquitectura, guía de incidentes, postmortems, despliegue,
    seguridad y preguntas frecuentes) las secciones más relevantes para la consulta. Devuelve documento_id,
    título y contenido de cada sección. Usala para saber qué pasó en un incidente, cuándo fue, a cuántos
    clientes afectó o qué dice un procedimiento."""
    resultados = await asyncio.to_thread(buscar_secciones, consulta, k)
    if not resultados:
        return {"error": f"No encontré nada para '{consulta}'. Probá con otras palabras clave."}
    return {"resultados": [{"documento_id": s.documento_id, "titulo": s.titulo, "contenido": s.contenido,
                            "puntaje": p} for s, p in resultados]}


@tool(args_schema=BuscarResenasInput)
async def buscar_resenas(desde: date, hasta: date, tema: str | None = None) -> dict[str, Any]:
    """Trae las reseñas de clientes de Colibrí publicadas entre dos fechas, inclusive: id, fecha, calificación
    (1 a 5 estrellas), canal y texto. Usala para saber qué opinan los clientes en un período. Para comparar
    dos momentos (por ejemplo antes y después de un incidente), llamala una vez por cada período."""
    resenas = await asyncio.to_thread(resenas_entre, desde, hasta, tema)
    return {"periodo": f"{desde.isoformat()}/{hasta.isoformat()}", "tema": tema, "cantidad": len(resenas),
            "resenas": resenas}


HERRAMIENTAS = [buscar_documentacion, buscar_resenas]


def extraer_hallazgos(mensajes: list, ya_vistos: set[str]) -> list[Hallazgo]:
    """Convierte las respuestas de las herramientas en Hallazgos validados (sin repetir referencias)."""
    hallazgos: list[Hallazgo] = []
    for herramienta, resultado in observaciones(mensajes):
        if not isinstance(resultado, dict) or "error" in resultado:
            continue
        try:
            if herramienta == "buscar_documentacion":
                nuevos = [Hallazgo(tipo="documento", referencia=r["documento_id"],
                                   contenido=f"{r['titulo']}\n{r['contenido']}") for r in resultado["resultados"]]
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
            if h.referencia not in ya_vistos:
                ya_vistos.add(h.referencia)
                hallazgos.append(h)
    return hallazgos


def crear_investigador(modelo: BaseChatModel):
    agente = create_react_agent(modelo, HERRAMIENTAS, prompt=PROMPT, name="investigador")

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
