"""El equipo de agentes de punta a punta, a través de la API: cola, workers, grafo, aprobación humana,
fallas y reinicios."""
import asyncio
import time

from app.config import ENVIOS
from app.graph import crear_modelos
from app.modelo_simulado import ModeloSimulado
from app.state import AccionPropuesta, DecisionSupervisor
from tests.ayudas import ANALISIS, CUPON_2, CUPON_5_DEMORADOS, CUPON_10, crear, esperar

RECORRIDO_ANALISIS = ["supervisor", "investigador", "supervisor", "investigador", "supervisor", "analista",
                      "supervisor", "redactor", "validador", "supervisor"]


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


def modelos_con(supervisor: ModeloSimulado) -> dict:
    return crear_modelos(sin_api=True) | {"supervisor": supervisor}


def test_el_serializador_de_redis_necesita_la_lista_de_clases(serde):
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
    accion = AccionPropuesta(tipo="enviar_cupones", segmento="cobro_doble", porcentaje=10, destinatarios=312,
                             costo_estimado=212160, descripcion="prueba")
    sin_lista = JsonPlusRedisSerializer()
    assert isinstance(sin_lista.loads_typed(sin_lista.dumps_typed(accion)), dict)   # el bug que evita la lista
    assert isinstance(serde.loads_typed(serde.dumps_typed(accion)), AccionPropuesta)


async def test_analisis_completo_con_rag(abrir_api):
    async with abrir_api() as cliente:
        job = await esperar(cliente, (await crear(cliente, ANALISIS))["job_id"], {"DONE", "FAILED"})
    assert job["estado"] == "DONE", job["error"]
    assert job["resultado"]["validado"]
    assert job["recorrido"] == RECORRIDO_ANALISIS
    assert "operador" not in job["recorrido"]   # sin acción pedida, no pasa por el operador ni por la aprobación
    informe = job["resultado"]["informe"]
    # Las fuentes del informe son documento_id reales de la base de conocimiento, traídos por el buscador híbrido
    assert any(f.startswith("postmortem_pagos_2026-10.md#") for f in informe["fuentes"])
    assert any(f.startswith("resenas ") for f in informe["fuentes"])
    assert job["resultado"]["turno"] == 1


async def test_aprobar_un_job_que_no_espera_da_409(abrir_api):
    async with abrir_api() as cliente:
        job_id = (await crear(cliente, ANALISIS))["job_id"]
        await esperar(cliente, job_id, {"DONE", "FAILED"})
        assert (await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})).status_code == 409


async def test_accion_cara_espera_aprobacion_y_se_ejecuta_una_vez(abrir_api, redis):
    async with abrir_api() as cliente:
        job_id = (await crear(cliente, CUPON_10))["job_id"]
        job = await esperar(cliente, job_id, {"WAITING_APPROVAL", "DONE", "FAILED"})
        assert job["estado"] == "WAITING_APPROVAL"
        assert "212.160" in job["pregunta"]["pregunta"]   # 312 clientes x $6.800 x 10%
        assert await redis.llen(ENVIOS) == 0            # antes de aprobar no se ejecutó nada

        r = await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "Ana", "comentario": "ok"})
        job = await esperar(cliente, job_id, {"DONE", "FAILED"})
    assert r.status_code == 202 and job["estado"] == "DONE"
    assert await redis.llen(ENVIOS) == 1
    assert "Se enviaron 312 cupones" in (job["resultado"]["informe"]["accion"] or "")
    assert "operador,aprobacion_humana,ejecutor" in ",".join(job["recorrido"])


async def test_accion_rechazada_no_se_ejecuta(abrir_api, redis):
    async with abrir_api() as cliente:
        job_id = (await crear(cliente, CUPON_5_DEMORADOS))["job_id"]
        await esperar(cliente, job_id, {"WAITING_APPROVAL"})
        await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "revisor": "Ana", "comentario": "muy caro"})
        job = await esperar(cliente, job_id, {"DONE", "FAILED"})
    assert job["estado"] == "DONE"
    assert await redis.llen(ENVIOS) == 0
    assert "rechazada" in job["resultado"]["resultado_accion"]


async def test_accion_barata_se_aprueba_sola(abrir_api):
    async with abrir_api() as cliente:
        job = await esperar(cliente, (await crear(cliente, CUPON_2))["job_id"], {"DONE", "FAILED", "WAITING_APPROVAL"})
    assert job["estado"] == "DONE"   # 2% = $42.432, debajo del umbral de $50.000
    assert "automáticamente" in job["resultado"]["resultado_accion"]


async def test_accion_rechazada_no_se_vuelve_a_proponer(abrir_api, redis):
    async with abrir_api(modelos_con(SupervisorInsistente(rol="supervisor"))) as cliente:
        job_id = (await crear(cliente, CUPON_5_DEMORADOS))["job_id"]
        await esperar(cliente, job_id, {"WAITING_APPROVAL"})
        await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "revisor": "Ana", "comentario": "no"})
        job = await esperar(cliente, job_id, {"DONE", "FAILED", "WAITING_APPROVAL"})
    assert job["estado"] == "DONE"
    assert job["recorrido"].count("aprobacion_humana") == 1
    assert await redis.llen(ENVIOS) == 0


async def test_cinco_pedidos_concurrentes(abrir_api):
    async with abrir_api() as cliente:
        inicio = time.perf_counter()
        respuestas = await asyncio.gather(*(cliente.post("/tasks", json={"pedido": ANALISIS}) for _ in range(5)))
        assert all(r.status_code == 202 for r in respuestas)
        assert time.perf_counter() - inicio < 1.0   # se aceptan al instante: nadie espera al agente
        finales = await asyncio.gather(*(esperar(cliente, r.json()["job_id"], {"DONE", "FAILED"}, 30) for r in respuestas))
    assert all(j["estado"] == "DONE" for j in finales)
    assert len({j["worker"] for j in finales}) > 1   # repartidos entre los workers


async def test_si_el_agente_falla_el_job_queda_en_failed(abrir_api):
    async with abrir_api(modelos_con(SupervisorRoto(rol="supervisor"))) as cliente:
        job = await esperar(cliente, (await crear(cliente, ANALISIS))["job_id"], {"DONE", "FAILED"})
    assert job["estado"] == "FAILED"
    assert "no responde" in job["error"]


async def test_despues_de_una_falla_los_workers_siguen(abrir_api):
    async with abrir_api(modelos_con(SupervisorRoto(rol="supervisor"))) as cliente:
        await esperar(cliente, (await crear(cliente, ANALISIS))["job_id"], {"DONE", "FAILED"})
    async with abrir_api() as cliente:
        job = await esperar(cliente, (await crear(cliente, ANALISIS))["job_id"], {"DONE", "FAILED"})
    assert job["estado"] == "DONE"


async def test_otra_instancia_retoma_un_job_pausado(abrir_api, redis):
    async with abrir_api() as cliente:
        job_id = (await crear(cliente, CUPON_10))["job_id"]
        await esperar(cliente, job_id, {"WAITING_APPROVAL"})
    # "Reinicio": se cierra la API y se abre otra con el mismo Redis y el mismo checkpointer
    async with abrir_api() as cliente:
        await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "Ana"})
        job = await esperar(cliente, job_id, {"DONE", "FAILED"})
    assert job["estado"] == "DONE"
    assert await redis.llen(ENVIOS) == 1
