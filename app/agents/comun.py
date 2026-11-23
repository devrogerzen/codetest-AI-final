"""Utilidades que comparten los nodos de los agentes."""
import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

log = logging.getLogger("orquestador")


def leer_json(contenido: Any) -> Any:
    try:
        return json.loads(contenido)
    except (TypeError, json.JSONDecodeError):
        return contenido


def observaciones(mensajes: list[BaseMessage]) -> list[tuple[str, Any]]:
    """(herramienta, resultado) de cada ToolMessage del ciclo ReAct de un especialista."""
    return [(m.name, leer_json(m.content)) for m in mensajes if isinstance(m, ToolMessage)]


def a_json(datos: Any) -> str:
    return json.dumps(datos, ensure_ascii=False, indent=1, default=str)


def siguiente_paso(state: dict) -> int:
    return len(state.get("aportes", [])) + 1
