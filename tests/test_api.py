"""La API: validación de entrada (Pydantic), códigos HTTP, RAG directo, streaming y API key."""
import json
import time

import app.main
from tests.ayudas import ANALISIS


async def test_health(abrir_api):
    async with abrir_api(workers=0) as cliente:
        salud = (await cliente.get("/health")).json()
    assert salud["redis"] is True
    assert salud["base_vectorial"]["fragmentos"] > 0
    assert salud["modelos"] == "simulados"


async def test_post_tasks_responde_202_al_instante(abrir_api, redis):
    async with abrir_api(workers=0) as cliente:
        inicio = time.perf_counter()
        r = await cliente.post("/tasks", json={"pedido": ANALISIS})
        ms = (time.perf_counter() - inicio) * 1000
    cuerpo = r.json()
    # No espera al agente (con workers=0 ni siquiera corre): el margen es para máquinas lentas o cargadas
    assert r.status_code == 202 and cuerpo["estado"] == "PENDING" and ms < 1000
    assert cuerpo["conversacion_id"] and cuerpo["consultar"] == f"/tasks/{cuerpo['job_id']}"
    assert await redis.exists(f"colibri:job:{cuerpo['job_id']}") == 1   # el estado se guardó antes de encolar


async def test_validacion_de_entrada(abrir_api):
    async with abrir_api(workers=0) as cliente:
        assert (await cliente.post("/tasks", json={"pedido": "hola"})).status_code == 422
        assert (await cliente.post("/tasks", json={"pedido": ANALISIS, "conversacion_id": "con espacios!"})).status_code == 422
        assert (await cliente.post("/tasks/x/approve", json={"aprobado": "quizás"})).status_code == 422
        assert (await cliente.post("/ask", json={"pregunta": "hola", "k": 50})).status_code == 422
        assert (await cliente.get("/tasks/no-existe")).status_code == 404


async def test_ask_responde_con_fuentes(abrir_api):
    async with abrir_api(workers=0) as cliente:
        r = await cliente.post("/ask", json={"pregunta": "¿Con qué comando se pasa Redis a la réplica?"})
        categoria_mala = await cliente.post("/ask", json={"pregunta": "¿Con qué comando se pasa Redis?", "categoria": "inexistente"})
    cuerpo = r.json()
    assert r.status_code == 200 and cuerpo["encontrada"]
    assert cuerpo["fuentes"][0]["documento_id"] == "guia_incidentes.md#runbook-redis-caido-o-inaccesible"
    assert set(f["chunk_id"] for f in cuerpo["fuentes"]) <= set(cuerpo["recuperados"])
    assert categoria_mala.status_code == 422


async def test_ask_stream_manda_eventos_sse(abrir_api):
    async with abrir_api(workers=0) as cliente:
        async with cliente.stream("POST", "/ask/stream", json={"pregunta": "¿Qué flag hay que activar si falla el proveedor de pagos?"}) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            eventos = [linea.removeprefix("event: ") async for linea in r.aiter_lines() if linea.startswith("event: ")]
            # (el cuerpo ya se consumió; los datos del fin se piden de nuevo para chequearlos)
        r2 = await cliente.post("/ask/stream", json={"pregunta": "¿Qué flag hay que activar si falla el proveedor de pagos?"})
    assert eventos[0] == "fuentes" and eventos[-1] == "fin" and "token" in eventos
    fin = json.loads(r2.text.strip().split("data: ")[-1])
    assert fin["encontrada"] and fin["fuentes_citadas"]


async def test_api_key_cuando_esta_configurada(abrir_api, monkeypatch):
    monkeypatch.setattr(app.main, "API_KEY", "secreta")
    async with abrir_api(workers=0) as cliente:
        sin_clave = await cliente.post("/tasks", json={"pedido": ANALISIS})
        con_clave = await cliente.post("/tasks", json={"pedido": ANALISIS}, headers={"X-API-Key": "secreta"})
        salud = await cliente.get("/health")   # el healthcheck no pide clave
    assert sin_clave.status_code == 401 and con_clave.status_code == 202 and salud.status_code == 200


async def test_openapi_documenta_los_endpoints(abrir_api):
    async with abrir_api(workers=0) as cliente:
        rutas = (await cliente.get("/openapi.json")).json()["paths"]
    assert {"/ask", "/ask/stream", "/tasks", "/tasks/{job_id}", "/tasks/{job_id}/approve",
            "/conversations/{conversacion_id}", "/health"} <= rutas.keys()
