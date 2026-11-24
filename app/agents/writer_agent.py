"""Redactor: la fase de síntesis. Escribe el informe final (salida estructurada) usando solo lo que
investigador y analista dejaron en el estado. Escribe únicamente en state["informe"].
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.comun import a_json, siguiente_paso
from app.config import TURNOS_DE_CONTEXTO
from app.state import Aporte, EstadoOrquestador, Informe

PROMPT = """Sos el redactor de un equipo de análisis de Colibrí. Escribís el informe final para el pedido \
del usuario usando SOLO los hallazgos y las métricas que te pasan.
- En "fuentes" van las referencias exactas de los hallazgos que usaste (al menos un documento y las reseñas, si las hay).
- En "metricas_citadas" van los nombres exactos de las métricas que mencionás.
- Todo número que escribas tiene que salir de una métrica o de un hallazgo. No calcules nada nuevo.
- Si hay resultado_accion, copialo tal cual en el campo "accion" (es lo que efectivamente pasó con la acción pedida).
- Si hay observaciones del validador, corregí exactamente eso.
- conversacion_previa es contexto de los pedidos anteriores: sirve para entender a qué se refiere el pedido, \
pero las fuentes y las métricas del informe son las de este turno.
Español claro, sin relleno. Las recomendaciones salen de los hallazgos (por ejemplo, de las acciones de un postmortem)."""


def crear_redactor(modelo: BaseChatModel):
    redactar = modelo.with_structured_output(Informe).with_retry(stop_after_attempt=2)

    async def redactor(state: EstadoOrquestador) -> dict:
        datos = {
            "pedido": state["pedido"],
            "instruccion": state["instruccion"],
            "conversacion_previa": [t.model_dump(exclude_none=True) for t in state.get("historial", [])[-TURNOS_DE_CONTEXTO:]],
            "hallazgos": [{"tipo": h.tipo, "referencia": h.referencia, "contenido": h.contenido} for h in state["hallazgos"]],
            "metricas": [m.model_dump() for m in state["metricas"]],
            "resultado_accion": state["resultado_accion"],
            "observaciones_del_validador": state["observaciones"],
        }
        informe = await redactar.ainvoke([SystemMessage(PROMPT), HumanMessage(f"Datos:\n{a_json(datos)}")])
        return {
            "informe": informe,
            "aportes": [Aporte(paso=siguiente_paso(state), agente="redactor",
                               resumen=f"Informe '{informe.titulo}' con {len(informe.fuentes)} fuentes y "
                                       f"{len(informe.metricas_citadas)} métricas citadas")],
            "messages": [AIMessage(f"{informe.titulo}: {informe.resumen}", name="redactor")],
        }

    return redactor
