"""Worker: saca mensajes de la cola de Redis y corre el grafo fuera del request HTTP.

Dos formas de correrlo:
- dentro del proceso de la API (WORKERS=3 en el .env, el default): tareas asyncio que arrancan con FastAPI;
- como proceso aparte, para escalar sin tocar la API (WORKERS=0 en la API y):  python -m app.worker --workers 4

Cada mensaje es {"job_id", "tipo": "iniciar"} o {"job_id", "tipo": "reanudar", "decision": {...}}.
"""
import argparse
import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from redis.asyncio import Redis

from app.config import RECURSION_LIMIT, REDIS_URL, SIN_API, resumir_error
from app.graph import construir_grafo, crear_modelos
from app.jobs import EstadoJob, RepositorioJobs
from app.observability import configurar_observabilidad, contexto_de_traza
from app.state import PERMITIDOS_JSON, estado_inicial

log = logging.getLogger("worker")


@asynccontextmanager
async def recursos(sin_api: bool = SIN_API) -> AsyncIterator[tuple[CompiledStateGraph, RepositorioJobs, Redis]]:
    """Abre Redis (jobs y cola), el checkpointer de LangGraph en Redis y compila el grafo."""
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
            # El serializador de Redis solo reconstruye clases permitidas: le pasamos las del estado
            checkpointer.serde = JsonPlusRedisSerializer(allowed_json_modules=PERMITIDOS_JSON)
            await checkpointer.asetup()   # crea los índices la primera vez
            grafo = construir_grafo(crear_modelos(sin_api), checkpointer, redis)
            yield grafo, RepositorioJobs(redis), redis
    finally:
        await redis.aclose()


def _resultado(valores: dict[str, Any]) -> dict[str, Any]:
    informe, accion = valores.get("informe"), valores.get("accion")
    return {
        "informe": informe.model_dump() if informe else None,
        "validado": valores.get("validado"),
        "observaciones": valores.get("observaciones", []),
        "accion": accion.model_dump() if accion else None,
        "resultado_accion": valores.get("resultado_accion"),
        "aportes": [a.model_dump() for a in valores.get("aportes", [])],
    }


async def procesar(grafo: CompiledStateGraph, repo: RepositorioJobs, mensaje: dict[str, Any], worker: str) -> EstadoJob | None:
    job_id = mensaje["job_id"]
    job = await repo.obtener(job_id)
    if job is None:
        log.warning("Mensaje para un job que no existe: %s", job_id)
        return None

    config = {"configurable": {"thread_id": job_id}, "recursion_limit": RECURSION_LIMIT}
    entrada = estado_inicial(job.pedido) if mensaje["tipo"] == "iniciar" else Command(resume=mensaje["decision"])
    await repo.actualizar(job_id, estado=EstadoJob.RUNNING, worker=worker)
    inicio, recorrido = time.perf_counter(), list(job.recorrido)

    def duracion() -> float:   # solo tiempo de proceso: no cuenta lo que esperó a la persona
        return round((job.duracion_s or 0) + time.perf_counter() - inicio, 3)

    try:
        with contexto_de_traza(job_id, mensaje["tipo"], worker):
            async for actualizacion in grafo.astream(entrada, config, stream_mode="updates"):
                nuevos = [nodo for nodo in actualizacion if nodo != "__interrupt__"]
                if nuevos:
                    recorrido += nuevos
                    await repo.actualizar(job_id, recorrido=recorrido)   # el progreso se ve por polling
        snapshot = await grafo.aget_state(config)
        if snapshot.next:   # quedó pausado en aprobacion_humana
            pregunta = snapshot.interrupts[0].value if snapshot.interrupts else {"pregunta": "Aprobación pendiente"}
            await repo.actualizar(job_id, estado=EstadoJob.WAITING_APPROVAL, pregunta=pregunta,
                                  recorrido=recorrido, duracion_s=duracion())
            log.info("%s: job %s esperando aprobación", worker, job_id[:8])
            return EstadoJob.WAITING_APPROVAL
        await repo.actualizar(job_id, estado=EstadoJob.DONE, resultado=_resultado(snapshot.values),
                              recorrido=recorrido, duracion_s=duracion())
        log.info("%s: job %s terminado en %.2f s", worker, job_id[:8], duracion())
        return EstadoJob.DONE
    except Exception as e:   # cualquier falla del agente deja el job en FAILED: nadie queda esperando para siempre
        log.exception("%s: job %s falló", worker, job_id[:8])
        await repo.actualizar(job_id, estado=EstadoJob.FAILED, error=resumir_error(e),
                              recorrido=recorrido, duracion_s=duracion())
        return EstadoJob.FAILED


async def correr_worker(nombre: str, grafo: CompiledStateGraph, repo: RepositorioJobs, detener: asyncio.Event) -> None:
    log.info("%s listo", nombre)
    while not detener.is_set():
        try:
            mensaje = await repo.siguiente_mensaje(timeout=1)
        except Exception as e:   # Redis caído: espera y reintenta en vez de morir
            log.warning("%s: no pude leer la cola (%s), reintento en 2 s", nombre, resumir_error(e))
            await asyncio.sleep(2)
            continue
        if mensaje:
            await procesar(grafo, repo, mensaje, nombre)


async def main(cantidad: int) -> None:
    configurar_observabilidad()
    detener = asyncio.Event()
    async with recursos() as (grafo, repo, _):
        tareas = [asyncio.create_task(correr_worker(f"worker-{i + 1}", grafo, repo, detener)) for i in range(cantidad)]
        try:
            await asyncio.gather(*tareas)
        finally:
            detener.set()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    argumentos = argparse.ArgumentParser(description="Workers del orquestador (proceso aparte de la API)")
    argumentos.add_argument("--workers", type=int, default=3)
    try:
        asyncio.run(main(argumentos.parse_args().workers))
    except KeyboardInterrupt:
        pass
