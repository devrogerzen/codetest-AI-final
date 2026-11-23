"""Pruebas sin infraestructura: la API completa (endpoints + workers + grafo + human-in-the-loop) contra
un Redis falso en memoria (fakeredis) y los modelos simulados. No hace falta Docker, Redis ni API key.

El checkpointer acá es MemorySaver porque fakeredis no implementa los módulos de Redis Stack que usa
RedisSaver, pero con el MISMO serializador que usa Redis (JSON + lista de clases permitidas): así un estado
que no sobreviva a Redis también falla acá. La persistencia real se prueba con la API levantada (README).

Uso:
    python prueba_sin_api.py
"""
import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager

os.environ.update(TRAZAS="0", SIN_API="1")   # antes de importar la app

import fakeredis   # noqa: E402
import httpx   # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer  # noqa: E402

from app.config import ENVIOS  # noqa: E402
from app.state import PERMITIDOS_JSON, AccionPropuesta, DecisionSupervisor  # noqa: E402
from app.graph import construir_grafo, crear_modelos  # noqa: E402
from app.jobs import RepositorioJobs  # noqa: E402
from app.main import crear_app  # noqa: E402
from app.modelo_simulado import ModeloSimulado  # noqa: E402

resultados: list[bool] = []
ANALISIS = "¿Cómo impactó el incidente de pagos de octubre en la opinión de los clientes? Quiero saber qué pasó."
CUPON_10 = "Mandá un cupón del 10% a los clientes con cobro doble por el incidente de pagos."
CUPON_5_DEMORADOS = "Compensá con un cupón del 5% a los clientes con el pedido demorado en el incidente de pagos."
CUPON_2 = "Mandá un cupón del 2% a los clientes con cobro doble."


def chequear(descripcion: str, condicion: bool, detalle: str = "") -> None:
    resultados.append(condicion)
    print(f"  {'OK   ' if condicion else 'FALLA'} {descripcion}{f'  ({detalle})' if detalle else ''}")


class SupervisorRoto(ModeloSimulado):
    """Falla siempre, como un LLM caído: el job tiene que terminar en FAILED, no colgado."""
    def _supervisor(self, estado: dict):
        raise RuntimeError("el proveedor del LLM no responde")


class SupervisorInsistente(ModeloSimulado):
    """Como Gemini en una corrida real: después de un rechazo vuelve a mandar la acción al operador."""
    def _supervisor(self, estado: dict):
        if estado["resultado_accion"] and not estado["informe"]:
            return DecisionSupervisor(siguiente="operador", instruccion="Volvé a proponer el cupón.", motivo="Insiste.")
        return super()._supervisor(estado)


def fabrica(redis, checkpointer, modelos=None):
    @asynccontextmanager
    async def recursos():
        grafo = construir_grafo(modelos or crear_modelos(sin_api=True), checkpointer, redis)
        yield grafo, RepositorioJobs(redis), redis
    return recursos


async def esperar(cliente: httpx.AsyncClient, job_id: str, estados: set[str], limite: float = 20) -> dict:
    inicio = time.perf_counter()
    while time.perf_counter() - inicio < limite:
        job = (await cliente.get(f"/tasks/{job_id}")).json()
        if job["estado"] in estados:
            return job
        await asyncio.sleep(0.05)
    return job


@asynccontextmanager
async def api(redis, checkpointer, modelos=None, workers: int = 3):
    app = crear_app(fabrica(redis, checkpointer, modelos), workers=workers)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as cliente:
            yield cliente


async def main() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    serde = JsonPlusRedisSerializer(allowed_json_modules=PERMITIDOS_JSON)
    memoria = MemorySaver(serde=serde)

    print("\n> Serialización de checkpoints (la misma que usa Redis)")
    accion = AccionPropuesta(tipo="enviar_cupones", segmento="cobro_doble", porcentaje=10, destinatarios=312,
                             costo_estimado=212160, descripcion="prueba")
    sin_lista = JsonPlusRedisSerializer()
    chequear("sin la lista de clases permitidas, un modelo del estado vuelve como dict (el bug que evita)",
             isinstance(sin_lista.loads_typed(sin_lista.dumps_typed(accion)), dict))
    chequear("con la lista, vuelve como AccionPropuesta", isinstance(serde.loads_typed(serde.dumps_typed(accion)), AccionPropuesta))

    async with api(redis, memoria) as cliente:
        print("\n> Endpoints")
        chequear("GET /health responde", (await cliente.get("/health")).json()["redis"] is True)
        inicio = time.perf_counter()
        r = await cliente.post("/tasks", json={"pedido": ANALISIS})
        ms = (time.perf_counter() - inicio) * 1000
        job_id = r.json()["job_id"]
        chequear("POST /tasks responde 202 con job_id y PENDING sin esperar al agente",
                 r.status_code == 202 and r.json()["estado"] == "PENDING" and ms < 300, f"{ms:.0f} ms")
        chequear("el job ya existe en Redis cuando el POST responde (estado antes que la cola)",
                 await redis.exists(f"colibri:job:{job_id}") == 1)
        chequear("un pedido demasiado corto se rechaza con 422 (validación Pydantic)",
                 (await cliente.post("/tasks", json={"pedido": "hola"})).status_code == 422)
        chequear("GET de un job que no existe da 404", (await cliente.get("/tasks/no-existe")).status_code == 404)

        print("\n> Worker y grafo")
        job = await esperar(cliente, job_id, {"DONE", "FAILED"})
        chequear("el análisis termina en DONE con el informe validado",
                 job["estado"] == "DONE" and job["resultado"]["validado"], job["estado"])
        chequear("el recorrido es el del orquestador (investigador x2, analista, redactor, validador)",
                 job["recorrido"] == ["supervisor", "investigador", "supervisor", "investigador", "supervisor", "analista",
                                      "supervisor", "redactor", "validador", "supervisor"], " -> ".join(job["recorrido"]))
        chequear("el pedido sin acción no pasa por el operador ni por la aprobación", "operador" not in job["recorrido"])
        chequear("aprobar un job que no está esperando da 409",
                 (await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})).status_code == 409)

        print("\n> Human-in-the-loop")
        critico = (await cliente.post("/tasks", json={"pedido": CUPON_10})).json()["job_id"]
        job = await esperar(cliente, critico, {"WAITING_APPROVAL", "DONE", "FAILED"})
        chequear("una acción cara (cupón del 10% a 312 clientes) pausa el grafo en WAITING_APPROVAL",
                 job["estado"] == "WAITING_APPROVAL", job["estado"])
        chequear("la pregunta muestra la acción y el costo estimado ($212.160)", "212.160" in job["pregunta"]["pregunta"],
                 job["pregunta"]["pregunta"][:80])
        chequear("antes de aprobar no se ejecutó nada (ningún envío registrado)", await redis.llen(ENVIOS) == 0)
        r = await cliente.post(f"/tasks/{critico}/approve", json={"aprobado": True, "revisor": "Ana", "comentario": "ok"})
        job = await esperar(cliente, critico, {"DONE", "FAILED"})
        chequear("al aprobar, el mismo hilo se reanuda y termina en DONE", r.status_code == 202 and job["estado"] == "DONE")
        chequear("la acción se ejecutó una sola vez", await redis.llen(ENVIOS) == 1)
        chequear("el informe dice qué pasó con la acción", "Se enviaron 312 cupones" in (job["resultado"]["informe"]["accion"] or ""),
                 job["resultado"]["resultado_accion"])
        chequear("el recorrido muestra operador -> aprobacion_humana -> ejecutor",
                 "operador,aprobacion_humana,ejecutor" in ",".join(job["recorrido"]))

        rechazado = (await cliente.post("/tasks", json={"pedido": CUPON_5_DEMORADOS})).json()["job_id"]
        await esperar(cliente, rechazado, {"WAITING_APPROVAL"})
        await cliente.post(f"/tasks/{rechazado}/approve", json={"aprobado": False, "revisor": "Ana", "comentario": "muy caro"})
        job = await esperar(cliente, rechazado, {"DONE", "FAILED"})
        chequear("si la persona rechaza, termina en DONE sin ejecutar la acción",
                 job["estado"] == "DONE" and await redis.llen(ENVIOS) == 1 and "rechazada" in job["resultado"]["resultado_accion"])

        barato = (await cliente.post("/tasks", json={"pedido": CUPON_2})).json()["job_id"]
        job = await esperar(cliente, barato, {"DONE", "FAILED", "WAITING_APPROVAL"})
        chequear("una acción barata (2% = $42.432, bajo el umbral) se aprueba sola, sin pausa",
                 job["estado"] == "DONE" and "automáticamente" in job["resultado"]["resultado_accion"], job["estado"])

        print("\n> Concurrencia")
        inicio = time.perf_counter()
        respuestas = await asyncio.gather(*(cliente.post("/tasks", json={"pedido": ANALISIS}) for _ in range(5)))
        ms = (time.perf_counter() - inicio) * 1000
        chequear("5 POST concurrentes se aceptan todos al instante", all(r.status_code == 202 for r in respuestas), f"{ms:.0f} ms los 5")
        finales = await asyncio.gather(*(esperar(cliente, r.json()["job_id"], {"DONE", "FAILED"}, 30) for r in respuestas))
        chequear("los 5 terminan en DONE, repartidos entre los workers",
                 all(j["estado"] == "DONE" for j in finales), ", ".join(sorted({j["worker"] for j in finales})))

    async with api(redis, memoria, crear_modelos(sin_api=True) | {"supervisor": SupervisorInsistente(rol="supervisor")}) as cliente:
        envios_antes = await redis.llen(ENVIOS)
        insistido = (await cliente.post("/tasks", json={"pedido": CUPON_5_DEMORADOS})).json()["job_id"]
        await esperar(cliente, insistido, {"WAITING_APPROVAL"})
        await cliente.post(f"/tasks/{insistido}/approve", json={"aprobado": False, "revisor": "Ana", "comentario": "no"})
        job = await esperar(cliente, insistido, {"DONE", "FAILED", "WAITING_APPROVAL"})
        chequear("una acción rechazada no se vuelve a proponer aunque el supervisor insista",
                 job["estado"] == "DONE" and job["recorrido"].count("aprobacion_humana") == 1
                 and await redis.llen(ENVIOS) == envios_antes, job["estado"])

    print("\n> Fallas y reinicios")
    async with api(redis, memoria, crear_modelos(sin_api=True) | {"supervisor": SupervisorRoto(rol="supervisor")}) as cliente:
        roto = (await cliente.post("/tasks", json={"pedido": ANALISIS})).json()["job_id"]
        job = await esperar(cliente, roto, {"DONE", "FAILED"})
        chequear("si el agente falla, el job queda en FAILED con el error (el cliente no espera para siempre)",
                 job["estado"] == "FAILED" and "no responde" in (job["error"] or ""), job["error"])
    async with api(redis, memoria) as cliente:
        job = await esperar(cliente, (await cliente.post("/tasks", json={"pedido": ANALISIS})).json()["job_id"], {"DONE", "FAILED"})
        chequear("después de una falla los workers siguen atendiendo", job["estado"] == "DONE")
        pausado = (await cliente.post("/tasks", json={"pedido": CUPON_10})).json()["job_id"]
        await esperar(cliente, pausado, {"WAITING_APPROVAL"})
    # La API "se reinicia": se cierra y se abre otra instancia con el mismo Redis y el mismo checkpointer
    async with api(redis, memoria) as cliente:
        await cliente.post(f"/tasks/{pausado}/approve", json={"aprobado": True, "revisor": "Ana"})
        job = await esperar(cliente, pausado, {"DONE", "FAILED"})
        chequear("otra instancia de la API retoma un job pausado y lo termina", job["estado"] == "DONE")

    print(f"\n{sum(resultados)}/{len(resultados)} chequeos correctos")
    raise SystemExit(0 if all(resultados) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
