"""Operador: el especialista que prepara acciones con efectos en el mundo real (por ejemplo, mandar
cupones de descuento a clientes).

Solo PROPONE: cotiza la acción y la deja en state["accion"]. No ejecuta nada. Ejecutar es trabajo
del nodo `ejecutor`, que corre después de la aprobación humana (ver app/hitl.py).
"""
import json
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, ValidationError

from app.agents.comun import log, observaciones, siguiente_paso
from app.config import RECURSION_LIMIT_ESPECIALISTA, SEGMENTOS
from app.state import AccionPropuesta, Aporte, EstadoOrquestador

PROMPT = """Sos el operador de un equipo de Colibrí. Preparás acciones sobre clientes, pero no las ejecutás: \
las cotizás para que alguien las apruebe.
- Usá cotizar_cupones con el segmento y el porcentaje que pide la tarea.
- El segmento sale del pedido del usuario: si habla de pedidos demorados, es pedidos_demorados; si habla de \
cobros dobles, es cobro_doble. Si la tarea dice otra cosa, manda el pedido del usuario.
- Si la tarea no dice el porcentaje, usá 10.
Al terminar, respondé en una línea qué acción quedó propuesta y su costo estimado."""


class CotizarCuponesInput(BaseModel):
    segmento: Literal["cobro_doble", "pedidos_demorados"] = Field(
        description="cobro_doble: clientes con cobro doble en el incidente de pagos; "
                    "pedidos_demorados: clientes con el pedido demorado en el incidente")
    porcentaje: int = Field(ge=1, le=50, description="Porcentaje de descuento del cupón, entre 1 y 50")


def _segmentos() -> dict[str, Any]:
    return json.loads(SEGMENTOS.read_text(encoding="utf-8"))


@tool(args_schema=CotizarCuponesInput)
async def cotizar_cupones(segmento: str, porcentaje: int) -> dict[str, Any]:
    """Cotiza el envío de un cupón de descuento a un segmento de clientes afectados por un incidente: cuántos
    clientes reciben el cupón y cuánto costaría (clientes x ticket promedio x porcentaje). No envía nada:
    solo prepara la acción para que la apruebe una persona."""
    datos = _segmentos()[segmento]
    costo = datos["clientes"] * datos["ticket_promedio"] * porcentaje / 100
    return {"tipo": "enviar_cupones", "segmento": segmento, "porcentaje": porcentaje,
            "destinatarios": datos["clientes"], "costo_estimado": round(costo, 2),
            "descripcion": f"Enviar un cupón del {porcentaje}% a {datos['clientes']} {datos['descripcion']}"}


def crear_operador(modelo: BaseChatModel):
    agente = create_react_agent(modelo, [cotizar_cupones], prompt=PROMPT, name="operador")

    async def operador(state: EstadoOrquestador) -> dict:
        tarea = HumanMessage(f"Pedido del usuario: {state['pedido']}\nTu tarea: {state['instruccion']}")
        resultado = await agente.ainvoke({"messages": [tarea]}, {"recursion_limit": RECURSION_LIMIT_ESPECIALISTA})
        accion = None
        for herramienta, datos in reversed(observaciones(resultado["messages"])):
            if herramienta == "cotizar_cupones" and isinstance(datos, dict):
                try:
                    accion = AccionPropuesta.model_validate(datos)
                    break
                except ValidationError as e:
                    log.warning("Acción descartada: %s", e)
        resumen = (f"Propone: {accion.descripcion} (costo estimado ${accion.costo_estimado:,.0f})".replace(",", ".")
                   if accion else "No pudo preparar ninguna acción.")
        return {
            "accion": accion,
            "aportes": [Aporte(paso=siguiente_paso(state), agente="operador", resumen=resumen)],
            "messages": [AIMessage(resumen, name="operador")],
        }

    return operador
