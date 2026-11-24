"""API de Colibrí Intelligence (FastAPI): RAG + equipo de agentes con supervisor + aprobación humana + trazas.

Dos formas de usarla:

- Preguntas puntuales a la base de conocimiento (RAG directo, responde en segundos):
    POST /ask                     -> respuesta + fuentes verificadas
    POST /ask/stream              -> la misma respuesta en Server-Sent Events, palabra por palabra

- Tareas complejas para el equipo de agentes (cola + workers: nunca bloquea el request):
    POST /tasks                   -> {"job_id", "conversacion_id", "estado": "PENDING"}  (202)
    GET  /tasks/{job_id}          -> estado, recorrido del grafo, pregunta de aprobación o resultado
    POST /tasks/{job_id}/approve  -> reanuda un job en WAITING_APPROVAL con la decisión de una persona
    GET  /tasks                   -> últimos jobs
    GET  /conversations/{id}      -> turnos de la conversación, leídos del checkpointer

    GET  /health                  -> Redis, base vectorial, workers y modo

Correr:  uvicorn app.main:app --port 8000      (documentación interactiva en /docs)
"""
import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from app.config import (API_KEY, PHOENIX_ENDPOINT, PROYECTO_TRAZAS, SIN_API, TRAZAS_ACTIVAS, WORKERS,
                        nombre_coleccion, proveedor_embeddings, resumir_error)
from app.jobs import ConversacionOcupada, EstadoJob, Job
from app.observability import configurar_observabilidad, contexto_de_traza
from app.rag.respuestas import Pregunta, RespuestaRAG, construir_cadena, responder_en_stream
from app.state import DecisionHumana, Turno
from app.worker import Recursos, correr_worker, recursos

log = logging.getLogger("api")

DESCRIPCION = """Sistema de IA de Colibrí: responde con la base de conocimiento de la empresa (RAG híbrido) y
resuelve tareas de varios pasos con un equipo de agentes coordinado por un supervisor. Las acciones caras
esperan la aprobación de una persona y cada ejecución queda trazada en Arize Phoenix."""


class NuevaTarea(BaseModel):
    pedido: str = Field(min_length=10, max_length=2000,
                        examples=["¿Cómo impactó el incidente de pagos de octubre en la opinión de los clientes?"])
    conversacion_id: str | None = Field(default=None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$",
                                        description="Para seguir una conversación: el id que devolvió un pedido anterior")


class TareaEncolada(BaseModel):
    job_id: str
    conversacion_id: str
    estado: EstadoJob
    consultar: str


class Conversacion(BaseModel):
    conversacion_id: str
    jobs: list[str]
    turnos: list[Turno]


def crear_app(fabrica_de_recursos: Callable[[], AbstractAsyncContextManager[Recursos]] = recursos,
              workers: int = WORKERS) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        configurar_observabilidad()
        async with fabrica_de_recursos() as r:
            app.state.recursos = r
            app.state.cadena_rag = construir_cadena(r.buscador, r.asistente)
            detener = asyncio.Event()
            tareas = [asyncio.create_task(correr_worker(f"api-worker-{i + 1}", r.grafo, r.repo, detener))
                      for i in range(workers)]
            yield
            detener.set()
            await asyncio.gather(*tareas, return_exceptions=True)

    async def verificar_api_key(clave: str | None = Depends(APIKeyHeader(name="X-API-Key", auto_error=False))) -> None:
        if API_KEY and clave != API_KEY:
            raise HTTPException(status_code=401, detail="Falta el header X-API-Key o no es válido")

    app = FastAPI(title="Colibrí Intelligence", version="2.0.0", description=DESCRIPCION, lifespan=lifespan)
    protegido = [Depends(verificar_api_key)]

    @app.exception_handler(ConversacionOcupada)
    async def conversacion_ocupada(request: Request, e: ConversacionOcupada) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(e), "job_activo": e.job_activo})

    def r(request: Request) -> Recursos:
        return request.app.state.recursos

    async def buscar(request: Request, job_id: str) -> Job:
        job = await r(request).repo.obtener(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No existe el job {job_id}")
        return job

    # --- RAG directo ---
    @app.post("/ask", response_model=RespuestaRAG, tags=["Base de conocimiento"], dependencies=protegido,
              summary="Pregunta puntual: RAG híbrido con fuentes verificadas")
    async def preguntar(pregunta: Pregunta, request: Request) -> RespuestaRAG:
        try:
            with contexto_de_traza(f"ask-{uuid.uuid4()}", "ask"):
                return await request.app.state.cadena_rag.ainvoke(pregunta.model_dump())
        except ValueError as e:   # categoría inexistente
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:   # el proveedor del LLM o de embeddings no responde
            log.exception("Falló /ask")
            raise HTTPException(status_code=503, detail=f"No pude responder: {resumir_error(e)}")

    @app.post("/ask/stream", tags=["Base de conocimiento"], dependencies=protegido,
              summary="La misma pregunta, en Server-Sent Events (fuentes, tokens y fin)",
              response_class=StreamingResponse)
    async def preguntar_en_stream(pregunta: Pregunta, request: Request) -> StreamingResponse:
        recursos_ = r(request)

        async def eventos() -> AsyncIterator[str]:
            with contexto_de_traza(f"ask-{uuid.uuid4()}", "ask_stream"):
                async for evento in responder_en_stream(recursos_.buscador, recursos_.asistente, pregunta):
                    yield evento

        return StreamingResponse(eventos(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # --- Tareas para el equipo de agentes ---
    @app.post("/tasks", status_code=202, response_model=TareaEncolada, tags=["Agentes"], dependencies=protegido,
              summary="Encola una tarea compleja (responde al instante)",
              responses={409: {"description": "La conversación todavía está procesando otro pedido"}})
    async def crear_tarea(tarea: NuevaTarea, request: Request) -> TareaEncolada:
        job = await r(request).repo.crear(tarea.pedido, tarea.conversacion_id)
        return TareaEncolada(job_id=job.id, conversacion_id=job.conversacion_id, estado=job.estado,
                             consultar=f"/tasks/{job.id}")

    @app.get("/tasks/{job_id}", response_model=Job, tags=["Agentes"], dependencies=protegido,
             summary="Estado de un job (polling)")
    async def consultar_tarea(job_id: str, request: Request) -> Job:
        return await buscar(request, job_id)

    @app.post("/tasks/{job_id}/approve", status_code=202, response_model=TareaEncolada, tags=["Agentes"],
              dependencies=protegido, summary="Aprueba o rechaza la acción de un job pausado")
    async def aprobar_tarea(job_id: str, decision: DecisionHumana, request: Request) -> TareaEncolada:
        job = await buscar(request, job_id)
        if job.estado != EstadoJob.WAITING_APPROVAL:
            raise HTTPException(status_code=409, detail=f"El job está en {job.estado}: solo se aprueba en WAITING_APPROVAL")
        await r(request).repo.encolar_reanudacion(job_id, decision.model_dump())
        return TareaEncolada(job_id=job_id, conversacion_id=job.conversacion_id, estado=EstadoJob.PENDING,
                             consultar=f"/tasks/{job_id}")

    @app.get("/tasks", response_model=list[Job], tags=["Agentes"], dependencies=protegido, summary="Últimos jobs")
    async def listar_tareas(request: Request, limite: int = 20) -> list[Job]:
        return await r(request).repo.recientes(limite)

    @app.get("/conversations/{conversacion_id}", response_model=Conversacion, tags=["Agentes"],
             dependencies=protegido, summary="Turnos de una conversación (memoria del checkpointer)")
    async def consultar_conversacion(conversacion_id: str, request: Request) -> Conversacion:
        recursos_ = r(request)
        jobs = await recursos_.repo.jobs_de_conversacion(conversacion_id)
        if not jobs:
            raise HTTPException(status_code=404, detail=f"No existe la conversación {conversacion_id}")
        estado = await recursos_.grafo.aget_state({"configurable": {"thread_id": conversacion_id}})
        return Conversacion(conversacion_id=conversacion_id, jobs=jobs, turnos=estado.values.get("historial", []))

    @app.get("/health", tags=["Sistema"], summary="Estado de las dependencias")
    async def salud(request: Request) -> dict:
        recursos_ = r(request)
        return {"redis": await recursos_.redis.ping(),
                "base_vectorial": {"coleccion": nombre_coleccion(), "fragmentos": len(recursos_.buscador.fragmentos),
                                   "embeddings": proveedor_embeddings()},
                "workers_en_api": workers, "modelos": "simulados" if SIN_API else "llm",
                "trazas": f"{PHOENIX_ENDPOINT} (proyecto {PROYECTO_TRAZAS})" if TRAZAS_ACTIVAS else "desactivadas"}

    return app


app = crear_app()
