"""Estado compartido del orquestador y los contratos (Pydantic) entre agentes.

Cada agente es dueño de su parcela del estado y solo escribe ahí:

    investigador -> hallazgos        analista -> metricas         redactor -> informe
    operador     -> accion           aprobacion -> decision_humana ejecutor -> resultado_accion
    validador    -> validado / observaciones                       supervisor -> siguiente / instruccion / delegaciones

Todos agregan una entrada en `aportes`, que es el registro de quién aportó qué y en qué paso.
Las listas usan el reducer `acumular`: cada nodo agrega, nadie pisa lo que escribió otro.

Conversaciones: el thread_id del checkpointer es el id de la conversación, así que el estado sobrevive entre
pedidos (y a un reinicio). Cada pedido nuevo empieza un turno: `estado_turno()` vacía lo propio del turno
(hallazgos, métricas, informe...) y deja `historial` y `messages`, que son la memoria de la conversación.
"""
import operator
from typing import Annotated, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

Decision = Literal["investigador", "analista", "operador", "redactor", "FINALIZAR"]
Agente = Literal["supervisor", "investigador", "analista", "operador", "aprobacion", "ejecutor", "redactor", "validador"]


class Hallazgo(BaseModel):
    """Lo que trae el investigador: una sección de la documentación o un lote de reseñas."""
    tipo: Literal["documento", "resenas"]
    referencia: str = Field(min_length=3, description="documento_id, o el período de las reseñas")
    contenido: str
    ids_resenas: list[int] = Field(default_factory=list)


class Metrica(BaseModel):
    """Lo que calcula el analista. El nombre lleva el grupo entre corchetes: promedio[antes]."""
    nombre: str = Field(min_length=3)
    valor: float
    detalle: str = ""


class AccionPropuesta(BaseModel):
    """Lo que propone el operador. Proponer no tiene efectos: ejecutar sí, y por eso pasa por aprobación."""
    tipo: Literal["enviar_cupones"]
    segmento: Literal["cobro_doble", "pedidos_demorados"]
    porcentaje: int = Field(ge=1, le=50)
    destinatarios: int = Field(ge=0)
    costo_estimado: float = Field(ge=0, description="Pesos: destinatarios x ticket promedio x porcentaje")
    descripcion: str


class DecisionHumana(BaseModel):
    aprobado: bool
    revisor: str = Field(default="sin nombre", max_length=60)
    comentario: str = Field(default="", max_length=500)


class Aporte(BaseModel):
    paso: int = Field(ge=1)
    agente: Agente
    resumen: str


class DecisionSupervisor(BaseModel):
    siguiente: Decision = Field(description="Quién trabaja ahora, o FINALIZAR si el informe ya fue validado")
    instruccion: str = Field(description="La tarea concreta para el especialista, en una o dos oraciones")
    motivo: str = Field(description="Por qué esa decisión, según la rúbrica")


class Informe(BaseModel):
    titulo: str
    resumen: str = Field(description="La respuesta al pedido en dos o tres oraciones")
    hallazgos: list[str] = Field(min_length=1)
    metricas_citadas: list[str] = Field(description="Nombres exactos de las métricas usadas, tal como están en el estado")
    fuentes: list[str] = Field(default_factory=list,
                               description="Referencias exactas de los hallazgos usados (vacía solo si no hay hallazgos)")
    recomendaciones: list[str] = Field(default_factory=list)
    accion: str | None = Field(default=None, description="Qué pasó con la acción pedida, si hubo una (copiá resultado_accion)")


class Turno(BaseModel):
    """Un pedido ya resuelto de la conversación: lo que ve el supervisor en el turno siguiente. Su referencia
    (conversacion#turno-N) es una fuente válida para el informe de un turno posterior."""
    referencia: str = ""
    pedido: str
    titulo: str
    resumen: str
    accion: str | None = None


def acumular(actual: list | None, nuevo: list | None) -> list:
    """operator.add, pero None vacía la lista: así un turno nuevo arranca sin los hallazgos del anterior."""
    if nuevo is None:
        return []
    return (actual or []) + nuevo


class EstadoOrquestador(MessagesState):
    pedido: str
    siguiente: Decision
    instruccion: str
    delegaciones: int
    hallazgos: Annotated[list[Hallazgo], acumular]
    metricas: Annotated[list[Metrica], acumular]
    accion: AccionPropuesta | None
    decision_humana: DecisionHumana | None
    resultado_accion: str | None
    informe: Informe | None
    validado: bool
    observaciones: list[str]
    aportes: Annotated[list[Aporte], acumular]
    historial: Annotated[list[Turno], operator.add]   # no se vacía: es la memoria de la conversación


# Clases que el checkpointer puede reconstruir al leer de Redis. Por seguridad, el serializador JSON solo revive
# tipos de una lista permitida, escritos como partes del módulo: ("app", "state", "Informe"). Sin esto, al
# reanudar un hilo (o al leer el estado final) estos objetos volverían como diccionarios.
MODELOS_DEL_ESTADO = (Hallazgo, Metrica, AccionPropuesta, DecisionHumana, Aporte, DecisionSupervisor, Informe, Turno)
PERMITIDOS_JSON = [(*modelo.__module__.split("."), modelo.__name__) for modelo in MODELOS_DEL_ESTADO]


def estado_turno(pedido: str) -> dict:
    """La entrada de cada pedido. None en las listas = vaciarlas (ver `acumular`); historial y messages
    se acumulan entre turnos."""
    return {"pedido": pedido, "messages": [("user", pedido)], "siguiente": "investigador", "instruccion": "",
            "delegaciones": 0, "hallazgos": None, "metricas": None, "accion": None, "decision_humana": None,
            "resultado_accion": None, "informe": None, "validado": False, "observaciones": [], "aportes": None,
            "historial": []}
