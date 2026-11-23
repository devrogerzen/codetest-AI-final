"""Human-in-the-loop: la aprobación humana antes de ejecutar una acción crítica.

    operador (propone) -> aprobacion_humana (pausa si es crítica) -> ejecutor (aplica o descarta) -> supervisor

- `aprobacion_humana` llama a interrupt(): el grafo se detiene y el checkpointer (Redis) guarda el estado
  bajo el thread_id del job. La API lo informa como WAITING_APPROVAL.
- POST /tasks/{id}/approve encola la decisión y el worker reanuda el mismo hilo con Command(resume=...).
- Al reanudar, el nodo se ejecuta de nuevo desde el principio. Por eso NO hay efectos antes del
  interrupt(): el envío real está en `ejecutor`, que corre una sola vez, después de la decisión.
"""
import json
from datetime import datetime

from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from redis.asyncio import Redis

from app.agents.comun import siguiente_paso
from app.config import ENVIOS, UMBRAL_APROBACION
from app.state import AccionPropuesta, Aporte, DecisionHumana, EstadoOrquestador


def pesos(monto: float) -> str:
    return "$" + f"{monto:,.0f}".replace(",", ".")


def es_critica(accion: AccionPropuesta) -> bool:
    """Crítica = cuesta más que el umbral. Por debajo, se aprueba sola (no tiene sentido frenar todo)."""
    return accion.costo_estimado > UMBRAL_APROBACION


async def aprobacion_humana(state: EstadoOrquestador) -> dict:
    accion = state["accion"]
    if accion is None:
        decision = DecisionHumana(aprobado=False, revisor="sistema", comentario="No había ninguna acción para aprobar.")
    elif not es_critica(accion):
        decision = DecisionHumana(aprobado=True, revisor="automático",
                                  comentario=f"Costo {pesos(accion.costo_estimado)}, por debajo del umbral de {pesos(UMBRAL_APROBACION)}.")
    else:
        # Pausa: el valor que se pase en Command(resume=...) es lo que devuelve interrupt()
        respuesta = interrupt({
            "pregunta": f"¿Aprobás esta acción? {accion.descripcion}. Costo estimado: {pesos(accion.costo_estimado)}.",
            "accion": accion.model_dump(),
            "umbral": UMBRAL_APROBACION,
        })
        decision = DecisionHumana.model_validate(respuesta)
    veredicto = "aprobada" if decision.aprobado else "rechazada"
    resumen = f"Acción {veredicto} por {decision.revisor}" + (f": {decision.comentario}" if decision.comentario else ".")
    return {
        "decision_humana": decision,
        "aportes": [Aporte(paso=siguiente_paso(state), agente="aprobacion", resumen=resumen)],
        "messages": [AIMessage(resumen, name="aprobacion")],
    }


def crear_ejecutor(redis: Redis):
    """El único nodo con efectos. Corre después de la decisión humana, una sola vez por acción."""

    async def ejecutor(state: EstadoOrquestador) -> dict:
        accion, decision = state["accion"], state["decision_humana"]
        if accion and decision and decision.aprobado:
            await redis.rpush(ENVIOS, json.dumps({
                "fecha": datetime.now().isoformat(timespec="seconds"), "accion": accion.model_dump(),
                "aprobado_por": decision.revisor, "pedido": state["pedido"]}, ensure_ascii=False))
            quien = ("aprobados automáticamente por estar debajo del umbral" if decision.revisor == "automático"
                     else f"aprobados por {decision.revisor}")
            resultado = (f"Se enviaron {accion.destinatarios} cupones del {accion.porcentaje}% "
                         f"(costo estimado {pesos(accion.costo_estimado)}), {quien}.")
        elif accion:
            resultado = f"No se envió nada: la acción fue rechazada por {decision.revisor if decision else 'nadie'}."
        else:
            resultado = "No se ejecutó ninguna acción."
        return {
            "resultado_accion": resultado,
            "aportes": [Aporte(paso=siguiente_paso(state), agente="ejecutor", resumen=resultado)],
            "messages": [AIMessage(resultado, name="ejecutor")],
        }

    return ejecutor
