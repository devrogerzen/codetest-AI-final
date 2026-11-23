"""Nodo de validación: revisa el informe contra el estado antes de que el supervisor pueda cerrar.

No usa LLM: son controles deterministas, así que no se le puede "convencer" de aprobar algo mal.
Si encuentra problemas, los deja en state["observaciones"] y el supervisor decide quién los corrige.
"""
import re

from langchain_core.messages import AIMessage

from app.agents.comun import siguiente_paso
from app.state import Aporte, EstadoOrquestador

NUMERO = re.compile(r"\d+(?:[.,]\d+)?")


def _numeros(texto: str) -> set[str]:
    return {n.replace(",", ".") for n in NUMERO.findall(texto)}


def _formatos(valor: float) -> set[str]:
    return {f"{v:g}" for v in (valor, abs(valor), round(valor, 1), round(abs(valor), 1), round(valor), round(abs(valor)))} \
        | {f"{valor:.1f}", f"{abs(valor):.1f}", f"{valor:.2f}", f"{abs(valor):.2f}"}


def problemas_del_informe(state: EstadoOrquestador) -> list[str]:
    informe = state["informe"]
    if informe is None:
        return ["No hay informe."]
    problemas: list[str] = []
    referencias = {h.referencia: h.tipo for h in state["hallazgos"]}
    nombres = {m.nombre for m in state["metricas"]}

    for fuente in informe.fuentes:
        if fuente not in referencias:
            problemas.append(f"La fuente '{fuente}' no está entre los hallazgos.")
    tipos_citados = {referencias[f] for f in informe.fuentes if f in referencias}
    for tipo in set(referencias.values()) - tipos_citados:
        problemas.append(f"El informe no cita ninguna fuente de tipo '{tipo}'.")

    for metrica in informe.metricas_citadas:
        if metrica not in nombres:
            problemas.append(f"La métrica '{metrica}' no existe en el estado.")
    if nombres and not informe.metricas_citadas:
        problemas.append("El informe no cita ninguna métrica del analista.")

    # Si se pidió una acción, el informe tiene que decir qué pasó con ella (y no otra cosa)
    resultado = state["resultado_accion"]
    if resultado and informe.accion != resultado:
        problemas.append(f"El campo 'accion' del informe tiene que ser exactamente el resultado de la acción: '{resultado}'")

    # Cada número del informe tiene que salir de una métrica, un hallazgo o la acción (los de un dígito se ignoran)
    permitidos = _numeros(state["pedido"])
    for h in state["hallazgos"]:
        permitidos |= _numeros(f"{h.referencia} {h.contenido}")
    for m in state["metricas"]:
        permitidos |= _formatos(m.valor) | _numeros(m.detalle)
    if state["accion"]:
        permitidos |= _numeros(state["accion"].descripcion) | _formatos(state["accion"].costo_estimado)
    permitidos |= _numeros(resultado or "")
    texto = " ".join([informe.resumen, *informe.hallazgos])
    inventados = sorted(n for n in _numeros(texto) - permitidos if len(n.split(".")[0]) > 1 or "." in n)
    if inventados:
        problemas.append(f"Estos números no salen de ninguna métrica ni hallazgo: {', '.join(inventados)}.")
    return problemas


async def validador(state: EstadoOrquestador) -> dict:
    problemas = problemas_del_informe(state)
    resumen = "Informe aprobado." if not problemas else "Informe rechazado: " + " ".join(problemas)
    return {
        "validado": not problemas,
        "observaciones": problemas,
        "aportes": [Aporte(paso=siguiente_paso(state), agente="validador", resumen=resumen)],
        "messages": [AIMessage(resumen, name="validador")],
    }
