"""El orquestador multi-agente: supervisor + especialistas, con RAG, acción crítica, aprobación humana y
checkpoints en Redis.

    START -> supervisor --(rutear)--> investigador ---------------------------------> supervisor
                        \\----------> analista -------------------------------------> supervisor
                         \\---------> operador -> aprobacion_humana -> ejecutor ------> supervisor
                          \\--------> redactor -> validador -------------------------> supervisor
                           \\-------> END (cuando el supervisor decide FINALIZAR)

El investigador consulta la base de conocimiento con el buscador híbrido (app/rag), que se crea una sola vez
por proceso y se le pasa al nodo por cierre (no viaja en el estado).

`aprobacion_humana` puede pausar el grafo con interrupt(): el estado queda en el checkpointer (Redis) bajo
el thread_id = id de la conversación, y se reanuda con Command(resume=decisión) desde otro request o tras
un reinicio.
"""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

from app.agents import (crear_analista, crear_investigador, crear_operador, crear_redactor, crear_supervisor,
                        validador)
from app.config import crear_llm
from app.hitl import aprobacion_humana, crear_ejecutor
from app.rag.buscador import BuscadorHibrido
from app.state import EstadoOrquestador

ROLES = ("supervisor", "investigador", "analista", "operador", "redactor")


def rutear(state: EstadoOrquestador) -> Literal["investigador", "analista", "operador", "redactor", "__end__"]:
    """Arista condicional del supervisor. El Literal le dice a LangGraph (y al diagrama) los destinos posibles."""
    return END if state["siguiente"] == "FINALIZAR" else state["siguiente"]


def crear_modelos(sin_api: bool = False) -> dict[str, BaseChatModel]:
    if sin_api:
        from app.modelo_simulado import ModeloSimulado
        return {rol: ModeloSimulado(rol=rol) for rol in ROLES}
    return {rol: crear_llm(rol) for rol in ROLES}


def construir_grafo(modelos: dict[str, BaseChatModel], checkpointer: BaseCheckpointSaver | None,
                    redis: Redis, buscador: BuscadorHibrido) -> CompiledStateGraph:
    grafo = StateGraph(EstadoOrquestador)
    grafo.add_node("supervisor", crear_supervisor(modelos["supervisor"]))
    grafo.add_node("investigador", crear_investigador(modelos["investigador"], buscador))
    grafo.add_node("analista", crear_analista(modelos["analista"]))
    grafo.add_node("operador", crear_operador(modelos["operador"]))
    grafo.add_node("aprobacion_humana", aprobacion_humana)
    grafo.add_node("ejecutor", crear_ejecutor(redis))   # Redis va por cierre, no en el estado
    grafo.add_node("redactor", crear_redactor(modelos["redactor"]))
    grafo.add_node("validador", validador)

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges("supervisor", rutear)
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")
    grafo.add_edge("operador", "aprobacion_humana")
    grafo.add_edge("aprobacion_humana", "ejecutor")
    grafo.add_edge("ejecutor", "supervisor")
    grafo.add_edge("redactor", "validador")
    grafo.add_edge("validador", "supervisor")
    return grafo.compile(checkpointer=checkpointer)
