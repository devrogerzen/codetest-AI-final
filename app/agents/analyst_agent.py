"""Agente de análisis: procesa las reseñas que trajo el investigador (sentimiento, estadísticas y
comparación entre grupos).

No busca información: trabaja con los ids que ya están en state["hallazgos"] y escribe
únicamente en state["metricas"].
"""
import statistics
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, ValidationError

from app.agents.comun import a_json, log, observaciones, siguiente_paso
from app.config import RECURSION_LIMIT_ESPECIALISTA
from app.datos import resenas_por_id, sentimiento
from app.state import Aporte, EstadoOrquestador, Metrica

PROMPT = """Sos el analista de un equipo de análisis de Colibrí. Procesás los datos que ya consiguió el \
investigador: no buscás información nueva.
- Usá los ids de reseñas tal como vienen en los grupos que te pasan; no inventes ids.
- analizar_sentimiento y estadisticas_calificaciones trabajan sobre un grupo; comparar_grupos, sobre dos.
- Podés pedir varias herramientas a la vez si no dependen entre sí.
Al terminar, respondé en dos o tres líneas con las métricas más importantes."""


class GrupoInput(BaseModel):
    ids: list[int] = Field(min_length=1, description="ids de las reseñas del grupo, tal como vienen en los hallazgos")
    etiqueta: str = Field(min_length=2, max_length=30, description="Nombre corto del grupo, por ejemplo 'antes' o 'después'")


class CompararInput(BaseModel):
    ids_a: list[int] = Field(min_length=1, description="ids del primer grupo (por ejemplo, antes del incidente)")
    ids_b: list[int] = Field(min_length=1, description="ids del segundo grupo (por ejemplo, después del incidente)")
    etiqueta_a: str = Field(min_length=2, max_length=30)
    etiqueta_b: str = Field(min_length=2, max_length=30)


def _metrica(nombre: str, valor: float, detalle: str = "") -> dict[str, Any]:
    return {"nombre": nombre, "valor": round(valor, 2), "detalle": detalle}


def _pct_negativas(resenas: list[dict]) -> float:
    return 100 * sum(sentimiento(r["texto"])[0] == "negativo" for r in resenas) / len(resenas)


@tool(args_schema=GrupoInput)
async def analizar_sentimiento(ids: list[int], etiqueta: str) -> dict[str, Any]:
    """Clasifica cada reseña del grupo como positiva, negativa o neutra (léxico en español que tiene en cuenta
    negaciones como 'sin problema') y devuelve el porcentaje de cada clase. Usala para medir cómo se sienten
    los clientes en un grupo de reseñas."""
    try:
        resenas = resenas_por_id(ids)
    except ValueError as e:
        return {"error": str(e)}
    etiquetas = {r["id"]: sentimiento(r["texto"])[0] for r in resenas}
    n = len(resenas)
    cuenta = {clase: sum(e == clase for e in etiquetas.values()) for clase in ("positivo", "negativo", "neutro")}
    return {"metricas": [
        _metrica(f"sentimiento_negativo_pct[{etiqueta}]", 100 * cuenta["negativo"] / n, f"{cuenta['negativo']} de {n} reseñas"),
        _metrica(f"sentimiento_positivo_pct[{etiqueta}]", 100 * cuenta["positivo"] / n, f"{cuenta['positivo']} de {n} reseñas"),
    ], "ids_negativos": [i for i, e in etiquetas.items() if e == "negativo"]}


@tool(args_schema=GrupoInput)
async def estadisticas_calificaciones(ids: list[int], etiqueta: str) -> dict[str, Any]:
    """Calcula estadísticas de las calificaciones (1 a 5 estrellas) de un grupo de reseñas: cantidad, promedio,
    mediana y porcentaje de calificaciones bajas (1 o 2 estrellas)."""
    try:
        notas = [r["calificacion"] for r in resenas_por_id(ids)]
    except ValueError as e:
        return {"error": str(e)}
    bajas = sum(n <= 2 for n in notas)
    return {"metricas": [
        _metrica(f"cantidad[{etiqueta}]", len(notas)),
        _metrica(f"promedio[{etiqueta}]", statistics.mean(notas), "calificación promedio, de 1 a 5"),
        _metrica(f"mediana[{etiqueta}]", statistics.median(notas)),
        _metrica(f"bajas_pct[{etiqueta}]", 100 * bajas / len(notas), f"{bajas} de {len(notas)} con 1 o 2 estrellas"),
    ]}


@tool(args_schema=CompararInput)
async def comparar_grupos(ids_a: list[int], ids_b: list[int], etiqueta_a: str, etiqueta_b: str) -> dict[str, Any]:
    """Compara dos grupos de reseñas, por ejemplo antes y después de un incidente: diferencia de la calificación
    promedio, variación porcentual del promedio y cambio en el porcentaje de reseñas negativas (en puntos
    porcentuales). Usala cuando el pedido pide medir un impacto o comparar dos momentos."""
    try:
        a, b = resenas_por_id(ids_a), resenas_por_id(ids_b)
    except ValueError as e:
        return {"error": str(e)}
    media_a = statistics.mean(r["calificacion"] for r in a)
    media_b = statistics.mean(r["calificacion"] for r in b)
    clave = f"{etiqueta_a}->{etiqueta_b}"
    return {"metricas": [
        _metrica(f"diferencia_promedio[{clave}]", media_b - media_a, f"{media_a:.2f} -> {media_b:.2f}"),
        _metrica(f"variacion_promedio_pct[{clave}]", 100 * (media_b - media_a) / media_a),
        _metrica(f"cambio_negativas_pp[{clave}]", _pct_negativas(b) - _pct_negativas(a), "puntos porcentuales"),
    ]}


HERRAMIENTAS = [analizar_sentimiento, estadisticas_calificaciones, comparar_grupos]


def extraer_metricas(mensajes: list, ya_vistas: set[str]) -> list[Metrica]:
    metricas: list[Metrica] = []
    for herramienta, resultado in observaciones(mensajes):
        if not isinstance(resultado, dict):
            continue
        for crudo in resultado.get("metricas", []):
            try:
                metrica = Metrica.model_validate(crudo)
            except ValidationError as e:
                log.warning("Métrica descartada de %s: %s", herramienta, e)
                continue
            if metrica.nombre not in ya_vistas:
                ya_vistas.add(metrica.nombre)
                metricas.append(metrica)
    return metricas


def crear_analista(modelo: BaseChatModel):
    agente = create_react_agent(modelo, HERRAMIENTAS, prompt=PROMPT, name="analista")

    async def analista(state: EstadoOrquestador) -> dict:
        # Solo le pasamos los grupos de reseñas (ids y período), no la documentación ni el historial
        grupos = [{"referencia": h.referencia, "cantidad": len(h.ids_resenas), "ids": h.ids_resenas}
                  for h in state["hallazgos"] if h.tipo == "resenas"]
        tarea = HumanMessage(f"Pedido del usuario: {state['pedido']}\nTu tarea: {state['instruccion']}\n"
                             f"Grupos de reseñas disponibles:\n{a_json(grupos)}")
        resultado = await agente.ainvoke({"messages": [tarea]}, {"recursion_limit": RECURSION_LIMIT_ESPECIALISTA})
        metricas = extraer_metricas(resultado["messages"], {m.nombre for m in state["metricas"]})
        resumen = resultado["messages"][-1].text or f"{len(metricas)} métricas nuevas."
        return {
            "metricas": metricas,
            "aportes": [Aporte(paso=siguiente_paso(state), agente="analista",
                               resumen=f"{resumen} [{len(metricas)} métricas]")],
            "messages": [AIMessage(resumen, name="analista")],
        }

    return analista
