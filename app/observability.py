"""Observabilidad con Arize Phoenix (OpenInference sobre OpenTelemetry).

`configurar_observabilidad()` registra el exportador hacia el colector de Phoenix e instrumenta
LangChain/LangGraph: cada nodo del grafo, cada llamada al LLM y cada herramienta queda como un span,
con sus tokens (y por lo tanto su costo) y su latencia. Se llama una sola vez al arrancar la API o el worker.

`contexto_de_traza(...)` agrega a todos los spans de un pedido el session.id (= id de la conversación) y
metadata útil: el job, la versión de los prompts, el tipo de mensaje y el worker. Así en Phoenix se filtra
por conversación, por job o por versión de prompts.
"""
import logging
from contextlib import contextmanager
from typing import Iterator

from app.config import PHOENIX_ENDPOINT, PROMPT_VERSION, PROYECTO_TRAZAS, TRAZAS_ACTIVAS

log = logging.getLogger("observabilidad")
_tracer = None


def configurar_observabilidad() -> None:
    global _tracer
    if _tracer is not None or not TRAZAS_ACTIVAS:
        return
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register
    # Sin proveedor global: así solo LangChain/LangGraph manda trazas. Si fuera global, FastAPI también
    # mandaría un span por cada GET de polling, y esas trazas sueltas ensuciarían las latencias del panel.
    proveedor = register(project_name=PROYECTO_TRAZAS, endpoint=PHOENIX_ENDPOINT, batch=True,
                         set_global_tracer_provider=False, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=proveedor)
    _tracer = proveedor.get_tracer("colibri")
    log.info("Trazas hacia Phoenix: %s (proyecto %s)", PHOENIX_ENDPOINT, PROYECTO_TRAZAS)


@contextmanager
def span_raiz(nombre: str, entrada: str) -> Iterator[None]:
    """Un span padre para operaciones que no son un único Runnable (el streaming: búsqueda + generación).
    Sin esto, cada parte sería una traza suelta en Phoenix."""
    if _tracer is None:
        yield
        return
    from openinference.instrumentation import get_attributes_from_context
    atributos = {"openinference.span.kind": "CHAIN", "input.value": entrada, **dict(get_attributes_from_context())}
    with _tracer.start_as_current_span(nombre, attributes=atributos):
        yield


@contextmanager
def contexto_de_traza(sesion: str, tipo: str, **metadata: str) -> Iterator[None]:
    """sesion = id de la conversación: en Phoenix, todos los turnos de una conversación quedan en la misma
    sesión. tipo = iniciar, reanudar o ask; metadata = job_id, worker, etc."""
    if not TRAZAS_ACTIVAS:
        yield
        return
    from openinference.instrumentation import using_attributes
    with using_attributes(session_id=sesion, tags=[tipo], prompt_template_version=PROMPT_VERSION,
                          metadata={"tipo": tipo, "prompt_version": PROMPT_VERSION, **metadata}):
        yield
