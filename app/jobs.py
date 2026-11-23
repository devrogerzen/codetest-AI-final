"""Estado de los trabajos (jobs) y cola de mensajes, los dos en Redis.

- Cada job es un hash `colibri:job:<id>` con su estado, el pedido, el resultado o el error.
- La cola es una lista `colibri:cola`: la API hace RPUSH y los workers BLPOP.
- Regla de oro: el estado se guarda ANTES de encolar. Si el worker fuera más rápido que la escritura,
  el cliente podría consultar un job que todavía no existe y recibir un 404.
"""
import json
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.config import COLA, PREFIJO_JOB


class EstadoJob(StrEnum):
    PENDING = "PENDING"                    # encolado, esperando un worker
    RUNNING = "RUNNING"                    # un worker está corriendo el grafo
    WAITING_APPROVAL = "WAITING_APPROVAL"  # pausado en el nodo de aprobación humana
    DONE = "DONE"
    FAILED = "FAILED"


class Job(BaseModel):
    id: str
    pedido: str
    estado: EstadoJob
    creado: str
    actualizado: str
    worker: str | None = None
    recorrido: list[str] = Field(default_factory=list)
    pregunta: dict[str, Any] | None = None   # lo que pregunta el nodo de aprobación humana
    decision: dict[str, Any] | None = None   # lo que respondió la persona
    resultado: dict[str, Any] | None = None
    error: str | None = None
    duracion_s: float | None = None          # tiempo de proceso, sin contar la espera humana


def _ahora() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


class RepositorioJobs:
    def __init__(self, redis: Redis):
        self.redis = redis

    def _clave(self, job_id: str) -> str:
        return f"{PREFIJO_JOB}{job_id}"

    async def crear(self, pedido: str) -> Job:
        ahora = _ahora()
        job = Job(id=str(uuid.uuid4()), pedido=pedido, estado=EstadoJob.PENDING, creado=ahora, actualizado=ahora)
        await self.guardar(job)                                                   # 1. primero el estado
        await self.redis.rpush(COLA, json.dumps({"job_id": job.id, "tipo": "iniciar"}))   # 2. después la cola
        return job

    async def guardar(self, job: Job) -> None:
        datos = {k: json.dumps(v, ensure_ascii=False) for k, v in job.model_dump(mode="json").items()}
        await self.redis.hset(self._clave(job.id), mapping=datos)

    async def obtener(self, job_id: str) -> Job | None:
        datos = await self.redis.hgetall(self._clave(job_id))
        if not datos:
            return None
        return Job.model_validate({k: json.loads(v) for k, v in datos.items()})

    async def actualizar(self, job_id: str, **campos: Any) -> None:
        campos["actualizado"] = _ahora()
        datos = {k: json.dumps(v.value if isinstance(v, EstadoJob) else v, ensure_ascii=False) for k, v in campos.items()}
        await self.redis.hset(self._clave(job_id), mapping=datos)

    async def encolar_reanudacion(self, job_id: str, decision: dict[str, Any]) -> None:
        await self.actualizar(job_id, estado=EstadoJob.PENDING, decision=decision, pregunta=None)
        await self.redis.rpush(COLA, json.dumps({"job_id": job_id, "tipo": "reanudar", "decision": decision}))

    async def siguiente_mensaje(self, timeout: float = 1.0) -> dict[str, Any] | None:
        """Bloquea hasta `timeout` segundos esperando un mensaje (BLPOP no consume CPU mientras espera)."""
        par = await self.redis.blpop([COLA], timeout=timeout)
        return json.loads(par[1]) if par else None

    async def recientes(self, limite: int = 20) -> list[Job]:
        jobs = []
        async for clave in self.redis.scan_iter(match=f"{PREFIJO_JOB}*", count=100):
            if job := await self.obtener(clave.removeprefix(PREFIJO_JOB)):
                jobs.append(job)
        return sorted(jobs, key=lambda j: j.creado, reverse=True)[:limite]
