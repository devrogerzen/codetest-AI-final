"""Modelos simulados para correr el orquestador sin API key (python main.py --sin-api).

Cada rol tiene reglas simples que leen lo mismo que leería el LLM real (el resumen del estado,
la instrucción del supervisor, los resultados de las herramientas) y devuelven lo mismo que
devolvería: una decisión estructurada, un pedido de herramientas o un informe. Sirven para probar
el grafo, las herramientas, la validación y los límites sin gastar tokens. Solo entienden pedidos
parecidos al de la demo: no reemplazan a un modelo real.
"""
import json
import re
import uuid
from datetime import date, timedelta
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.agents.comun import leer_json
from app.rag.respuestas import NO_LO_SE, RespuestaLLM
from app.state import DecisionSupervisor, Informe
from app.texto import normalizar, tokenizar

FECHA = re.compile(r"(\d{2})/(\d{2})/(\d{4})")


def _fechas(texto: str) -> list[date]:
    return [date(int(a), int(m), int(d)) for d, m, a in FECHA.findall(texto)]


def _dmy(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _num(valor: float) -> str:
    return f"{abs(valor):.2f}".rstrip("0").rstrip(".").replace(".", ",")


def _llamar(*llamadas: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": nombre, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}",
                                              "type": "tool_call"} for nombre, args in llamadas])


def _ultimo_humano(mensajes: Any) -> str:
    return next(m.text for m in reversed(list(mensajes)) if isinstance(m, HumanMessage))


def _mensajes(entrada: Any) -> list[BaseMessage]:
    """Lo que llega a with_structured_output: una lista de mensajes o el valor de un prompt."""
    return entrada.to_messages() if hasattr(entrada, "to_messages") else list(entrada)


def _datos(texto: str) -> Any:
    inicio = min((i for i in (texto.find("{"), texto.find("[")) if i >= 0), default=-1)
    return json.loads(texto[inicio:]) if inicio >= 0 else {}


class ModeloSimulado(BaseChatModel):
    rol: str

    @property
    def _llm_type(self) -> str:
        return f"simulado-{self.rol}"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ModeloSimulado":
        return self   # las reglas ya conocen las herramientas de su rol

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        if schema is RespuestaLLM:
            return RunnableLambda(lambda entrada: self._asistente(_ultimo_humano(_mensajes(entrada))))
        reglas = {DecisionSupervisor: self._supervisor, Informe: self._redactor}[schema]
        return RunnableLambda(lambda mensajes: reglas(_datos(_ultimo_humano(_mensajes(mensajes)))))

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        if self.rol == "asistente":   # respuesta en texto (la usa el streaming de /ask/stream)
            salida = self._asistente(_ultimo_humano(messages))
            texto = salida.respuesta + "".join(f" [{f}]" for f in salida.fuentes)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(texto))])
        reglas = {"investigador": self._investigador, "analista": self._analista, "operador": self._operador}[self.rol]
        return ChatResult(generations=[ChatGeneration(message=reglas(messages))])

    # --- asistente de /ask: responde con la primera oración del fragmento que más se parece a la pregunta ---
    def _asistente(self, texto: str) -> RespuestaLLM:
        contexto, _, pregunta = texto.partition("PREGUNTA:")
        terminos = set(tokenizar(pregunta))
        mejor, coincidencias = None, 0
        for bloque in contexto.removeprefix("CONTEXTO:").split("\n\n---\n\n"):
            cabecera, _, cuerpo = bloque.strip().partition("\n")
            if not cabecera.startswith("["):
                continue
            comunes = len(terminos & set(tokenizar(cuerpo)))
            if comunes > coincidencias:
                mejor, coincidencias = (cabecera.strip("[]"), cuerpo), comunes
        if mejor is None or coincidencias < 2:
            return RespuestaLLM(respuesta=NO_LO_SE, encontrada_en_contexto=False, fuentes=[])
        chunk_id, cuerpo = mejor
        lineas = [linea for linea in cuerpo.splitlines()[1:] if linea.strip()]
        primera = re.split(r"(?<=[.:])\s", lineas[0] if lineas else cuerpo)[0]
        return RespuestaLLM(respuesta=primera, encontrada_en_contexto=True, fuentes=[chunk_id])

    # --- supervisor: aplica la rúbrica del prompt, en orden ---
    def _supervisor(self, estado: dict) -> DecisionSupervisor:
        pedido = normalizar(estado["pedido"])
        docs = [h for h in estado["hallazgos"] if h["tipo"] == "documento"]
        lotes = [h for h in estado["hallazgos"] if h["tipo"] == "resenas"]
        compara = any(p in pedido for p in ("impact", "compar", "antes", "cambi"))
        analiza = compara or any(p in pedido for p in ("opin", "resen"))
        pide_accion = any(p in pedido for p in ("cupon", "compens"))
        fechas = [f for h in docs for f in _fechas(h.get("extracto", ""))]
        if estado["informe"] and estado["validado"]:
            return DecisionSupervisor(siguiente="FINALIZAR", instruccion="", motivo="El informe está validado (paso 7).")
        if estado["informe"] and estado["observaciones_del_validador"]:
            return DecisionSupervisor(siguiente="redactor", motivo="El validador rechazó el informe (paso 6).",
                                      instruccion="Corregí el informe: " + " ".join(estado["observaciones_del_validador"]))
        if not docs:
            extra = " y traé las reseñas de clientes posteriores al incidente" if analiza else " y a qué clientes afectó"
            return DecisionSupervisor(siguiente="investigador", motivo="Falta el contexto del pedido (paso 1).",
                                      instruccion="Buscá en la documentación qué pasó en el incidente de pagos de octubre "
                                                  f"(fecha, duración e impacto){extra}.")
        if compara and len(lotes) < 2:
            inicio = min(fechas) if fechas else date(2026, 10, 14)
            return DecisionSupervisor(
                siguiente="investigador", motivo="Hay reseñas de un solo período y el pedido pide comparar (paso 2).",
                instruccion=f"Para medir el impacto falta la base de comparación: traé las reseñas del mes anterior al "
                            f"incidente, del {_dmy(inicio - timedelta(days=30))} al {_dmy(inicio - timedelta(days=1))}.")
        if analiza and not estado["metricas"]:
            return DecisionSupervisor(siguiente="analista", motivo="Faltan las métricas (paso 3).",
                                      instruccion="Calculá sentimiento y calificaciones de cada período y compará "
                                                  "antes contra después del incidente.")
        if pide_accion and not estado["resultado_accion"]:
            porcentaje = re.search(r"(\d+)\s*%", estado["pedido"])
            segmento = "clientes con el pedido demorado" if "demor" in pedido else "clientes con cobro doble"
            return DecisionSupervisor(siguiente="operador", motivo="El pedido pide una acción y todavía no se resolvió (paso 4).",
                                      instruccion=f"Prepará el envío de un cupón del {porcentaje[1] if porcentaje else 10}% "
                                                  f"a los {segmento}.")
        if not estado["informe"]:
            return DecisionSupervisor(siguiente="redactor", motivo="Ya está todo lo que pide el pedido (paso 5).",
                                      instruccion="Escribí el informe: qué pasó, qué cambió y qué se hizo.")
        return DecisionSupervisor(siguiente="FINALIZAR", instruccion="", motivo="No queda nada pendiente.")

    # --- operador: cotiza la acción que le piden (no ejecuta nada) ---
    def _operador(self, mensajes: list[BaseMessage]) -> AIMessage:
        tarea = next(m.text for m in mensajes if isinstance(m, HumanMessage)).split("Tu tarea:")[-1]
        obs = [leer_json(m.content) for m in mensajes if isinstance(m, ToolMessage)]
        if not obs:
            porcentaje = re.search(r"(\d+)\s*%", tarea)
            segmento = "pedidos_demorados" if "demor" in normalizar(tarea) else "cobro_doble"
            return _llamar(("cotizar_cupones", {"segmento": segmento,
                                                "porcentaje": int(porcentaje[1]) if porcentaje else 10}))
        datos = obs[-1]
        return AIMessage(f"Propuesta lista: {datos['descripcion']}, costo estimado ${datos['costo_estimado']:,.0f}."
                         .replace(",", "."))

    # --- investigador: documentación primero; si le piden reseñas sin fechas, las saca del documento ---
    def _investigador(self, mensajes: list[BaseMessage]) -> AIMessage:
        tarea = next(m.text for m in mensajes if isinstance(m, HumanMessage))
        tarea_norm = normalizar(tarea.split("Tu tarea:")[-1])
        obs = [(m.name, leer_json(m.content)) for m in mensajes if isinstance(m, ToolMessage)]
        usadas = {nombre for nombre, _ in obs}
        fechas = _fechas(tarea.split("Tu tarea:")[-1])
        if not obs:
            if "documentacion" in tarea_norm:
                return _llamar(("buscar_documentacion", {"consulta": "postmortem incidente pagos impacto clientes acciones", "k": 4}))
            if len(fechas) >= 2:
                return _llamar(("buscar_resenas", {"desde": fechas[0].isoformat(), "hasta": fechas[1].isoformat()}))
            return AIMessage("No entendí qué tengo que buscar: necesito un tema o un período.")
        if "buscar_resenas" not in usadas and "resenas" in tarea_norm:
            textos = " ".join(r["contenido"] for n, d in obs if n == "buscar_documentacion" for r in d.get("resultados", []))
            if fechas_doc := _fechas(textos):
                inicio = min(fechas_doc)   # la fecha del incidente es la primera que menciona el postmortem
                return _llamar(("buscar_resenas", {"desde": inicio.isoformat(),
                                                   "hasta": (inicio + timedelta(days=30)).isoformat()}))
        partes = []
        for nombre, datos in obs:
            if nombre == "buscar_documentacion" and "resultados" in datos:
                partes.append(f"{len(datos['resultados'])} secciones de documentación")
            elif nombre == "buscar_resenas":
                partes.append(f"{datos['cantidad']} reseñas del período {datos['periodo']}")
        return AIMessage("Encontré " + " y ".join(partes) + "." if partes else "No encontré información.")

    # --- analista: si hay dos grupos, todas las métricas de una vez (llamadas en paralelo) ---
    def _analista(self, mensajes: list[BaseMessage]) -> AIMessage:
        tarea = next(m.text for m in mensajes if isinstance(m, HumanMessage))
        grupos = sorted(_datos(tarea.split("Grupos de reseñas disponibles:")[-1]), key=lambda g: g["referencia"])
        obs = [leer_json(m.content) for m in mensajes if isinstance(m, ToolMessage)]
        if not obs:
            if len(grupos) >= 2:
                antes, despues = grupos[0]["ids"], grupos[-1]["ids"]
                return _llamar(("analizar_sentimiento", {"ids": antes, "etiqueta": "antes"}),
                               ("analizar_sentimiento", {"ids": despues, "etiqueta": "después"}),
                               ("estadisticas_calificaciones", {"ids": antes, "etiqueta": "antes"}),
                               ("estadisticas_calificaciones", {"ids": despues, "etiqueta": "después"}),
                               ("comparar_grupos", {"ids_a": antes, "ids_b": despues,
                                                    "etiqueta_a": "antes", "etiqueta_b": "después"}))
            if grupos:
                return _llamar(("analizar_sentimiento", {"ids": grupos[0]["ids"], "etiqueta": "período"}),
                               ("estadisticas_calificaciones", {"ids": grupos[0]["ids"], "etiqueta": "período"}))
            return AIMessage("No hay reseñas para analizar.")
        metricas = {m["nombre"]: m["valor"] for d in obs if isinstance(d, dict) for m in d.get("metricas", [])}
        clave = ("promedio[antes]", "promedio[después]")
        if all(c in metricas for c in clave):
            return AIMessage(f"Calculé {len(metricas)} métricas. El promedio pasó de {metricas[clave[0]]} a "
                             f"{metricas[clave[1]]} y las reseñas negativas, de {metricas['sentimiento_negativo_pct[antes]']}% "
                             f"a {metricas['sentimiento_negativo_pct[después]']}%.")
        return AIMessage(f"Calculé {len(metricas)} métricas.")

    # --- redactor: arma el informe solo con lo que hay en el estado ---
    def _redactor(self, datos: dict) -> Informe:
        m = {x["nombre"]: x["valor"] for x in datos["metricas"]}
        docs = [h for h in datos["hallazgos"] if h["tipo"] == "documento" and h["referencia"].startswith("postmortem")] \
            or [h for h in datos["hallazgos"] if h["tipo"] == "documento"]
        lotes = [h["referencia"] for h in datos["hallazgos"] if h["tipo"] == "resenas"]
        texto_docs = "\n".join(h["contenido"] for h in docs)
        fecha = _dmy(min(_fechas(texto_docs))) if _fechas(texto_docs) else "octubre"
        impacto = [re.sub(r"\*\*", "", linea[2:]).strip() for linea in texto_docs.splitlines() if linea.startswith("- ")]
        acciones = [fila.split("|")[1].strip() for fila in texto_docs.splitlines()
                    if fila.startswith("| ") and not fila.startswith("| Acción")]
        usadas = [n for n in ("promedio[antes]", "promedio[después]", "variacion_promedio_pct[antes->después]",
                              "sentimiento_negativo_pct[antes]", "sentimiento_negativo_pct[después]",
                              "bajas_pct[antes]", "bajas_pct[después]") if n in m]
        accion = datos.get("resultado_accion")
        if len(usadas) < 5:   # pedido sin análisis de reseñas: el informe cuenta el contexto y la acción
            return Informe(
                titulo=f"Compensación a clientes afectados por el incidente de pagos del {fecha}" if accion
                else f"Incidente de pagos del {fecha}",
                resumen=(f"Por el incidente de pagos del {fecha} se pidió compensar a los clientes afectados. {accion}"
                         if accion else "No hay métricas de reseñas para este pedido; se resume el incidente."),
                hallazgos=impacto[:3] or ["Sin hallazgos"], metricas_citadas=usadas,
                fuentes=[h["referencia"] for h in docs] + lotes, recomendaciones=acciones[:4], accion=accion)
        return Informe(
            accion=accion,
            titulo=f"Impacto del incidente de pagos del {fecha} en la opinión de los clientes",
            resumen=(f"Después del incidente del {fecha}, la calificación promedio de las reseñas bajó de "
                     f"{_num(m['promedio[antes]'])} a {_num(m['promedio[después]'])} (una caída del "
                     f"{_num(m['variacion_promedio_pct[antes->después]'])}%) y las reseñas negativas pasaron del "
                     f"{_num(m['sentimiento_negativo_pct[antes]'])}% al {_num(m['sentimiento_negativo_pct[después]'])}%. "
                     f"Las quejas se concentran en los cobros dobles y en la demora de las confirmaciones."),
            hallazgos=[*impacto[:3],
                       f"Las reseñas con 1 o 2 estrellas pasaron del {_num(m['bajas_pct[antes]'])}% al "
                       f"{_num(m['bajas_pct[después]'])}%."],
            metricas_citadas=usadas,
            fuentes=[h["referencia"] for h in docs] + lotes,
            recomendaciones=acciones[:4],
        )
