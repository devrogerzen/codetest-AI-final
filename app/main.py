"""API de producción del orquestador multi-agente de Colibrí (FastAPI).

Ningún endpoint corre el agente: POST /tasks guarda el job en Redis, lo encola y responde al instante con
un job_id (202 Accepted). El trabajo lo hacen los workers; el cliente consulta el estado por polling.

    POST /tasks                 -> {"job_id", "estado": "PENDING"}
    GET  /tasks/{job_id}        -> estado, recorrido del grafo, pregunta de aprobación o resultado
    POST /tasks/{job_id}/approve-> reanuda un job en WAITING_APPROVAL con la decisión de una persona
    GET  /tasks                 -> últimos jobs
    GET  /health                -> Redis, workers y modo

Correr:  uvicorn app.main:app --port 8000
"""
import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import PHOENIX_ENDPOINT, PROYECTO_TRAZAS, SIN_API, TRAZAS_ACTIVAS, WORKERS
from app.jobs import EstadoJob, Job, RepositorioJobs
from app.observability import configurar_observabilidad
from app.state import DecisionHumana
from app.worker import correr_worker, recursos


class NuevaTarea(BaseModel):
    pedido: str = Field(min_length=10, max_length=2000,
                        examples=["¿Cómo impactó el incidente de pagos de octubre en la opinión de los clientes?"])


class TareaEncolada(BaseModel):
    job_id: str
    estado: EstadoJob
    consultar: str


def crear_app(fabrica_de_recursos: Callable[[], AbstractAsyncContextManager] = recursos, workers: int = WORKERS) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        configurar_observabilidad()
        async with fabrica_de_recursos() as (grafo, repo, redis):
            app.state.repo, app.state.redis = repo, redis
            detener = asyncio.Event()
            tareas = [asyncio.create_task(correr_worker(f"api-worker-{i + 1}", grafo, repo, detener)) for i in range(workers)]
            yield
            detener.set()
            await asyncio.gather(*tareas, return_exceptions=True)

    app = FastAPI(title="Colibrí · Orquestador multi-agente", version="1.0.0", lifespan=lifespan)

    def repo(request: Request) -> RepositorioJobs:
        return request.app.state.repo

    async def buscar(request: Request, job_id: str) -> Job:
        job = await repo(request).obtener(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No existe el job {job_id}")
        return job

    @app.post("/tasks", status_code=202, response_model=TareaEncolada)
    async def crear_tarea(tarea: NuevaTarea, request: Request) -> TareaEncolada:
        job = await repo(request).crear(tarea.pedido)
        return TareaEncolada(job_id=job.id, estado=job.estado, consultar=f"/tasks/{job.id}")

    @app.get("/tasks/{job_id}", response_model=Job)
    async def consultar_tarea(job_id: str, request: Request) -> Job:
        return await buscar(request, job_id)

    @app.post("/tasks/{job_id}/approve", status_code=202, response_model=TareaEncolada)
    async def aprobar_tarea(job_id: str, decision: DecisionHumana, request: Request) -> TareaEncolada:
        job = await buscar(request, job_id)
        if job.estado != EstadoJob.WAITING_APPROVAL:
            raise HTTPException(status_code=409, detail=f"El job está en {job.estado}: solo se aprueba en WAITING_APPROVAL")
        await repo(request).encolar_reanudacion(job_id, decision.model_dump())
        return TareaEncolada(job_id=job_id, estado=EstadoJob.PENDING, consultar=f"/tasks/{job_id}")

    @app.get("/tasks", response_model=list[Job])
    async def listar_tareas(request: Request, limite: int = 20) -> list[Job]:
        return await repo(request).recientes(limite)

    @app.get("/health")
    async def salud(request: Request) -> dict:
        return {"redis": await request.app.state.redis.ping(), "workers_en_api": workers,
                "modelos": "simulados" if SIN_API else "llm",
                "trazas": f"{PHOENIX_ENDPOINT} (proyecto {PROYECTO_TRAZAS})" if TRAZAS_ACTIVAS else "desactivadas"}

    return app


app = crear_app()
