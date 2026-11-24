"""Supervisor: el router del sistema. Decide quién trabaja en cada paso y cuándo terminar.

No investiga, no calcula y no redacta: solo lee un resumen del estado y devuelve una
DecisionSupervisor (salida estructurada con un Literal de los nodos posibles).
"""
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.comun import a_json, siguiente_paso
from app.config import MAX_DELEGACIONES, TURNOS_DE_CONTEXTO
from app.state import Aporte, DecisionSupervisor, EstadoOrquestador, Turno
from app.texto import normalizar

PROMPT = """Sos el supervisor de un equipo de análisis de Colibrí, una tienda online. No investigás, no \
calculás y no redactás: decidís quién trabaja ahora y le das una instrucción concreta.

Especialistas:
- investigador: busca en la base de conocimiento (incidentes, postmortems, runbooks, arquitectura, seguridad) y trae reseñas de clientes por período.
- analista: calcula sobre las reseñas que ya trajo el investigador (sentimiento, estadísticas, comparación entre grupos). No busca nada.
- operador: prepara acciones sobre clientes (por ejemplo, enviar cupones de descuento). Solo las propone: después \
pasan solas por la aprobación humana y por el ejecutor, y el resultado vuelve como resultado_accion.
- redactor: escribe el informe final con los hallazgos, las métricas y el resultado de la acción, si hubo.

Rúbrica, en orden. Delegá al primero que no se cumpla:
1. Hay documentación que explique el contexto del pedido -> si no, investigador.
2. Si el pedido habla de impacto, cambios o comparación, hay reseñas de DOS períodos (antes y después) -> si no, investigador, con las fechas exactas que faltan.
3. Si el pedido pide analizar opiniones o reseñas, el analista calculó métricas sobre ellas, incluida la comparación entre períodos -> si no, analista.
4. Si el pedido pide ejecutar una acción (mandar cupones, compensar clientes) y todavía no hay resultado_accion -> operador, con el segmento y el porcentaje. \
Preguntar qué hacer o pedir recomendaciones NO es pedir una acción: las recomendaciones van en el informe y el operador no interviene.
5. Hay un informe -> si no, redactor.
6. El validador aprobó el informe -> si marcó observaciones, redactor (o el especialista al que le falte algo), pasándole las observaciones.
7. Todo se cumple -> FINALIZAR.

Nunca mandes dos veces al operador por la misma acción: si ya hay resultado_accion (aprobada o rechazada), seguí con el informe.

No repitas una delegación que ya se hizo con la misma instrucción. Las fechas van en formato dd/mm/aaaa.

Conversación: si el pedido se apoya en un turno anterior ("esos clientes", "lo que me dijiste", "y ahora..."), \
usá conversacion_previa para que la instrucción al especialista sea concreta (fechas, segmento, porcentaje). \
Lo que ya se respondió en un turno anterior no hace falta volver a investigarlo si el pedido no lo pide."""


ACCION = re.compile(r"\b(cupon\w*|compens\w*|descuento\w*|reintegr\w*|bonific\w*|mand\w*|envi[aáe]\w*|ejecut\w*)\b")


def pide_accion(pedido: str) -> bool:
    """¿El pedido pide hacer algo sobre clientes? Es un filtro conservador: deja pasar cualquier pedido que
    nombre una acción (después decide el LLM y, si es cara, una persona) y frena los que solo preguntan."""
    return bool(ACCION.search(normalizar(pedido)))


PORCENTAJE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")


def porcentajes_sin_fundamento(instruccion: str, state: EstadoOrquestador) -> list[str]:
    """Los porcentajes que el supervisor le pide al operador y que no salen del pedido, de lo investigado en
    este turno ni de un turno anterior: un dato que el LLM "recordó" y nadie verificó."""
    textos = [state["pedido"], *(h.contenido for h in state["hallazgos"]),
              *(f"{t.resumen} {t.accion or ''}" for t in state.get("historial", []))]
    conocidos = {p.replace(",", ".") for texto in textos for p in PORCENTAJE.findall(texto)}
    return [p for p in PORCENTAJE.findall(instruccion) if p.replace(",", ".") not in conocidos]


def contexto(state: EstadoOrquestador) -> dict:
    """Lo que ve el supervisor: un resumen del estado, no los mensajes internos de cada especialista."""
    return {
        "pedido": state["pedido"],
        "conversacion_previa": [t.model_dump(exclude_none=True) for t in state.get("historial", [])[-TURNOS_DE_CONTEXTO:]],
        "delegaciones": f"{state['delegaciones']} de {MAX_DELEGACIONES}",
        "aportes": [f"{a.paso}. {a.agente}: {a.resumen}" for a in state["aportes"]],
        "hallazgos": [{"tipo": h.tipo, "referencia": h.referencia,
                       **({"cantidad_resenas": len(h.ids_resenas)} if h.tipo == "resenas" else {"extracto": h.contenido[:300]})}
                      for h in state["hallazgos"]],
        "metricas": [f"{m.nombre} = {m.valor}" for m in state["metricas"]],
        "accion_propuesta": state["accion"].descripcion if state["accion"] else None,
        "resultado_accion": state["resultado_accion"],
        "informe": state["informe"].titulo if state["informe"] else None,
        "validado": state["validado"],
        "observaciones_del_validador": state["observaciones"],
    }


def crear_supervisor(modelo: BaseChatModel):
    decidir = modelo.with_structured_output(DecisionSupervisor).with_retry(stop_after_attempt=2)

    async def supervisor(state: EstadoOrquestador) -> dict:
        if state["delegaciones"] >= MAX_DELEGACIONES:
            # Condición de parada dura: pase lo que pase, se cierra con lo que haya
            decision = DecisionSupervisor(
                siguiente="FINALIZAR" if state["informe"] else "redactor",
                instruccion="" if state["informe"] else "Escribí el informe con lo disponible e indicá qué faltó.",
                motivo=f"Se alcanzó el límite de {MAX_DELEGACIONES} delegaciones.")
        else:
            decision = await decidir.ainvoke([SystemMessage(PROMPT),
                                              HumanMessage(f"Estado actual:\n{a_json(contexto(state))}")])
            if decision.siguiente == "FINALIZAR" and not state["validado"]:
                decision = DecisionSupervisor(
                    siguiente="redactor", motivo="No se puede finalizar sin un informe validado.",
                    instruccion="Escribí (o corregí) el informe teniendo en cuenta las observaciones del validador.")
            elif decision.siguiente == "operador" and state["resultado_accion"]:
                # La decisión humana es final: una acción ya aprobada o rechazada no se vuelve a proponer
                decision = DecisionSupervisor(
                    siguiente="redactor", motivo="La acción ya tiene una decisión humana: no se vuelve a proponer.",
                    instruccion="Escribí el informe e incluí en 'accion' el resultado de la acción tal como figura en el estado.")
            elif (decision.siguiente == "operador" and porcentajes_sin_fundamento(decision.instruccion, state)
                  and not any(h.tipo == "documento" for h in state["hallazgos"])):
                # Los parámetros de una acción se verifican antes de cotizarla: primero se busca el dato
                decision = DecisionSupervisor(
                    siguiente="investigador",
                    motivo="La acción usa un porcentaje que no sale del pedido ni de lo investigado: primero hay que buscarlo.",
                    instruccion="Antes de proponer la acción, buscá en la documentación los datos que necesita (por ejemplo, "
                                f"qué cupón recomienda el postmortem). Pedido: {state['pedido']}")
            elif decision.siguiente == "operador" and not pide_accion(state["pedido"]):
                # Mínimo privilegio: si el pedido solo pregunta, ninguna acción sobre clientes, aunque el LLM la sugiera
                decision = DecisionSupervisor(
                    siguiente="redactor" if state["hallazgos"] else "investigador",
                    motivo="El pedido no pide ejecutar ninguna acción: las recomendaciones van en el informe.",
                    instruccion="Incluí las acciones sugeridas como recomendaciones del informe, sin ejecutarlas."
                    if state["hallazgos"] else decision.instruccion)

        texto = f"-> {decision.siguiente}: {decision.instruccion}".rstrip(": ")
        cambios = {
            "siguiente": decision.siguiente,
            "instruccion": decision.instruccion,
            "aportes": [Aporte(paso=siguiente_paso(state), agente="supervisor", resumen=f"{texto} ({decision.motivo})")],
            "messages": [AIMessage(texto, name="supervisor")],
        }
        if decision.siguiente != "FINALIZAR":
            cambios["delegaciones"] = state["delegaciones"] + 1
        elif state["informe"]:
            # El turno queda en la memoria de la conversación (el checkpointer la guarda en Redis)
            informe = state["informe"]
            numero = len(state.get("historial", [])) + 1
            cambios["historial"] = [Turno(referencia=f"conversacion#turno-{numero}", pedido=state["pedido"],
                                          titulo=informe.titulo, resumen=informe.resumen,
                                          accion=state["resultado_accion"])]
        return cambios

    return supervisor
