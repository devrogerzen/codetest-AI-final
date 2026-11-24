"""Pruebas del sistema completo contra la API levantada: 9 ejecuciones que quedan trazadas en Phoenix.

  RAG directo (/ask)
    1. una pregunta que está en la base de conocimiento -> respuesta con fuente verificada
    2. una pregunta que NO está -> "No lo sé." (defensa contra la alucinación)
    3. la misma idea en streaming (/ask/stream) -> mide el tiempo hasta la primera palabra
  Equipo de agentes (/tasks, todas a la vez, como una carga real)
    4. análisis de impacto del incidente: RAG + reseñas + analista + redactor + validador
    5. cupón del 10% a los clientes con cobro doble -> pide aprobación humana -> se aprueba
    6. cupón del 5% a los pedidos demorados -> pide aprobación humana -> se rechaza (muy caro)
    7. cupón del 2% -> debajo del umbral, se aprueba solo
    8-9. una conversación de dos turnos: el segundo pedido ("mandales el cupón que recomienda el postmortem")
         solo se entiende con la memoria del primero, que vive en el checkpointer de Redis

Uso (con el sistema levantado: ./run.sh o docker compose up -d):
    python scripts/pruebas_sistema.py
    python scripts/pruebas_sistema.py --url http://127.0.0.1:8000 --salida pruebas_sistema.json

Se usa 127.0.0.1 y no "localhost": en Windows, "localhost" prueba primero IPv6 y cada conexión nueva
tarda ~300 ms en caer a IPv4. Eso es del cliente, no de la API.
"""
import argparse
import asyncio
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

PEDIDOS = [
    ("analisis", "¿Cómo impactó el incidente de pagos de octubre en la opinión de los clientes? Quiero saber qué pasó, "
                 "cuánto cambiaron las reseñas y qué deberíamos hacer."),
    ("cupon_aprobado", "Mandá un cupón del 10% a los clientes con cobro doble por el incidente de pagos."),
    ("cupon_rechazado", "Compensá con un cupón del 5% a los clientes con el pedido demorado en el incidente de pagos."),
    ("cupon_automatico", "Mandá un cupón del 2% a los clientes con cobro doble."),
]
CONVERSACION = [
    "¿Qué pasó en el incidente de pagos de octubre y a cuántos clientes afectó?",
    "Mandales a los clientes con cobro doble el cupón que recomienda el postmortem.",
]
PREGUNTAS = [
    ("rag_encontrada", "¿Qué tiene que activar la guardia si el proveedor de pagos responde con errores?"),
    ("rag_no_lo_se", "¿Cuál es la capital de Australia?"),
]
TERMINALES = {"DONE", "FAILED"}
REVISOR = "Roger (revisión de prueba)"


def decidir(pedido: str, pregunta: str) -> dict:
    """Lo que haría un revisor atento con la acción que propone el agente para este pedido."""
    texto = pedido.lower()
    if "cupón" not in texto and "compens" not in texto:
        return {"aprobado": False, "revisor": REVISOR, "comentario": "Este pedido no pidió ninguna compensación."}
    if "demorado" in texto:
        return {"aprobado": False, "revisor": REVISOR, "comentario": "Muy caro para este segmento: no lo apruebo."}
    if "postmortem" in texto and "cupón del 10%" not in pregunta:
        return {"aprobado": False, "revisor": REVISOR, "comentario": "El postmortem recomienda un cupón del 10%, no ese."}
    return {"aprobado": True, "revisor": REVISOR, "comentario": "Ok, es lo que pide el pedido."}


def percentil(valores: list[float], p: float) -> float:
    """Percentil por rango más cercano (con pocas muestras, el p95 es el valor más alto)."""
    ordenados = sorted(valores)
    return ordenados[max(0, math.ceil(p / 100 * len(ordenados)) - 1)]


async def preguntar(cliente: httpx.AsyncClient, nombre: str, pregunta: str) -> dict:
    inicio = time.perf_counter()
    r = await cliente.post("/ask", json={"pregunta": pregunta})
    r.raise_for_status()
    cuerpo = r.json()
    return {"prueba": nombre, "pregunta": pregunta, "segundos": round(time.perf_counter() - inicio, 2),
            "respuesta": cuerpo["respuesta"], "encontrada": cuerpo["encontrada"],
            "fuentes": [f["documento_id"] for f in cuerpo["fuentes"]]}


async def preguntar_en_stream(cliente: httpx.AsyncClient, pregunta: str) -> dict:
    inicio, primera, texto, fin = time.perf_counter(), None, "", {}
    async with cliente.stream("POST", "/ask/stream", json={"pregunta": pregunta}) as r:
        evento = None
        async for linea in r.aiter_lines():
            if linea.startswith("event: "):
                evento = linea.removeprefix("event: ")
            elif linea.startswith("data: ") and evento == "token":
                primera = primera or round(time.perf_counter() - inicio, 2)
                texto += json.loads(linea.removeprefix("data: "))
            elif linea.startswith("data: ") and evento == "fin":
                fin = json.loads(linea.removeprefix("data: "))
    return {"prueba": "rag_stream", "pregunta": pregunta, "primera_palabra_s": primera,
            "segundos": round(time.perf_counter() - inicio, 2), "respuesta": texto.strip(), **fin}


async def crear(cliente: httpx.AsyncClient, nombre: str, pedido: str, conversacion_id: str | None = None) -> dict:
    inicio = time.perf_counter()
    cuerpo = {"pedido": pedido} | ({"conversacion_id": conversacion_id} if conversacion_id else {})
    r = await cliente.post("/tasks", json=cuerpo)
    r.raise_for_status()
    return {**r.json(), "prueba": nombre, "pedido": pedido, "post_ms": round((time.perf_counter() - inicio) * 1000, 1),
            "t0": time.perf_counter()}


async def seguir(cliente: httpx.AsyncClient, job: dict, limite_s: float) -> dict:
    """Polling hasta que el job termina; si pide aprobación, decide como un revisor."""
    while time.perf_counter() - job["t0"] < limite_s:
        await asyncio.sleep(1)
        estado = (await cliente.get(f"/tasks/{job['job_id']}")).json()
        if estado["estado"] == "WAITING_APPROVAL":
            decision = decidir(job["pedido"], estado["pregunta"]["pregunta"])
            print(f"[{job['prueba']}] pide aprobación: {estado['pregunta']['pregunta']}")
            print(f"[{job['prueba']}] -> {'APRUEBO' if decision['aprobado'] else 'RECHAZO'}: {decision['comentario']}")
            job["decision"] = decision
            (await cliente.post(f"/tasks/{job['job_id']}/approve", json=decision)).raise_for_status()
        elif estado["estado"] in TERMINALES:
            job.update(final=estado, total_s=round(time.perf_counter() - job["t0"], 2))
            return job
    job["final"] = {"estado": "TIMEOUT"}
    return job


async def conversacion(cliente: httpx.AsyncClient, limite_s: float) -> list[dict]:
    primero = await seguir(cliente, await crear(cliente, "conversacion_turno_1", CONVERSACION[0]), limite_s)
    segundo = await crear(cliente, "conversacion_turno_2", CONVERSACION[1], primero["conversacion_id"])
    return [primero, await seguir(cliente, segundo, limite_s)]


async def main(url: str, limite_s: float, salida: str) -> int:
    async with httpx.AsyncClient(base_url=url, timeout=120) as cliente:
        salud = (await cliente.get("/health")).json()
        print(f"API: {url} · modelos: {salud['modelos']} · base vectorial: {salud['base_vectorial']}\n")

        print("> RAG directo")
        preguntas = [await preguntar(cliente, n, p) for n, p in PREGUNTAS]
        preguntas.append(await preguntar_en_stream(cliente, "¿Con qué comando se pasa Redis a la réplica?"))
        for p in preguntas:
            print(f"  {p['prueba']:<15} {p['segundos']:>5} s  {p['respuesta'].splitlines()[0][:90]}")

        print("\n> Equipo de agentes: 4 pedidos a la vez + una conversación de dos turnos")
        jobs = await asyncio.gather(*(crear(cliente, n, p) for n, p in PEDIDOS))
        print(f"  POST /tasks respondió en: {', '.join(f'{j['post_ms']} ms' for j in jobs)}\n")
        resultados = await asyncio.gather(*(seguir(cliente, j, limite_s) for j in jobs), conversacion(cliente, limite_s))
        jobs = [*resultados[:-1], *resultados[-1]]
        memoria = (await cliente.get(f"/conversations/{jobs[-1]['conversacion_id']}")).json()

    print(f"\n{'prueba':<22} {'estado':<7} {'POST':>8} {'proceso':>8} {'total':>7}  pasos  validado")
    for j in jobs:
        final = j["final"]
        print(f"{j['prueba']:<22} {final.get('estado'):<7} {j['post_ms']:>6} ms {final.get('duracion_s') or 0:>6} s "
              f"{j.get('total_s', 0):>5} s  {len(final.get('recorrido', [])):>5}  {str((final.get("resultado") or {}).get("validado")):>8}")
        if (final.get("resultado") or {}).get("resultado_accion"):
            print(f"{'':<22} acción: {final['resultado']['resultado_accion']}")
        if final.get("error"):
            print(f"{'':<22} error: {final['error']}")
    print(f"\nConversación {memoria['conversacion_id'][:8]}: {len(memoria['turnos'])} turnos en el checkpointer")
    for t in memoria["turnos"]:
        print(f"  - {t['pedido']}\n    -> {t['resumen'][:150]}")

    post = [j["post_ms"] for j in jobs]
    proceso = [j["final"]["duracion_s"] for j in jobs if j["final"].get("duracion_s")]
    print(f"\nRespuesta de la API (POST /tasks): p50 {percentil(post, 50)} ms · p95 {percentil(post, 95)} ms")
    if proceso:
        print(f"Tiempo de proceso del grafo:      p50 {percentil(proceso, 50)} s · p95 {percentil(proceso, 95)} s")
    print("Costo por ejecución, tokens y latencia por traza: en Phoenix (http://127.0.0.1:6006).")

    destino = Path(__file__).resolve().parent.parent / "resultados" / salida
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps({"fecha": datetime.now().isoformat(timespec="seconds"), "url": url, "salud": salud,
                                   "preguntas": preguntas, "jobs": [{k: v for k, v in j.items() if k != "t0"} for j in jobs],
                                   "conversacion": memoria}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detalle guardado en resultados/{salida}")
    ok = (all(j["final"]["estado"] == "DONE" and j["final"]["resultado"]["validado"] for j in jobs)
          and len(memoria["turnos"]) == 2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    argumentos = argparse.ArgumentParser(description="Pruebas del sistema completo contra la API")
    argumentos.add_argument("--url", default="http://127.0.0.1:8000")
    argumentos.add_argument("--limite", type=float, default=900, help="segundos máximos por job")
    argumentos.add_argument("--salida", default="pruebas_sistema.json")
    a = argumentos.parse_args()
    raise SystemExit(asyncio.run(main(a.url, a.limite, a.salida)))
