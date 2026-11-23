"""Observabilidad con Arize Phoenix (OpenInference sobre OpenTelemetry).

`configurar_observabilidad()` registra el exportador hacia el colector de Phoenix e instrumenta
LangChain/LangGraph: cada nodo del grafo, cada llamada al LLM y cada herramienta queda como un span,
con sus tokens (y por lo tanto su costo) y su latencia. Se llama una sola vez al arrancar la API o el worker.

`contexto_de_traza(job)` agrega a todos los spans de un job el session.id (= job_id) y metadata útil:
la versión de los prompts, el tipo de mensaje y el worker. Así en Phoenix se filtra por job o por versión.
"""
import logging
from contextlib import contextmanager
from typing import Iterator

from app.config import PHOENIX_ENDPOINT, PROMPT_VERSION, PROYECTO_TRAZAS, TRAZAS_ACTIVAS

log = logging.getLogger("observabilidad")
_configurado = False


def configurar_observabilidad() -> None:
    global _configurado
    if _configurado or not TRAZAS_ACTIVAS:
        return
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register
    # Sin proveedor global: así solo LangChain/LangGraph manda trazas. Si fuera global, FastAPI también
    # mandaría un span por cada GET de polling, y esas trazas sueltas ensuciarían las latencias del panel.
    proveedor = register(project_name=PROYECTO_TRAZAS, endpoint=PHOENIX_ENDPOINT, batch=True,
                         set_global_tracer_provider=False, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=proveedor)
    _configurado = True
    log.info("Trazas hacia Phoenix: %s (proyecto %s)", PHOENIX_ENDPOINT, PROYECTO_TRAZAS)


@contextmanager
def contexto_de_traza(job_id: str, tipo: str, worker: str) -> Iterator[None]:
    if not TRAZAS_ACTIVAS:
        yield
        return
    from openinference.instrumentation import using_attributes
    with using_attributes(session_id=job_id, tags=[tipo], prompt_template_version=PROMPT_VERSION,
                          metadata={"job_id": job_id, "tipo": tipo, "worker": worker, "prompt_version": PROMPT_VERSION}):
        yield
