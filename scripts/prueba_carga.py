"""Prueba de carga: 5 pedidos concurrentes contra la API, con polling y aprobación humana.

- Manda los 5 POST /tasks al mismo tiempo y mide cuánto tarda la API en responder (tiene que ser inmediato).
- Hace polling a GET /tasks/{id}. Cuando un job queda en WAITING_APPROVAL, decide como lo haría un revisor:
  aprueba lo que el pedido pidió y rechaza el cupón caro de los pedidos demorados y cualquier acción que el
  pedido no haya pedido. Decide por pedido y no por orden de llegada, porque con un LLM real el orden cambia.
- Al final imprime una tabla y los percentiles p50 / p95, y guarda el detalle en resultados/ (--salida).

Uso (con la API corriendo en otra terminal):
    python scripts/prueba_carga.py
    python scripts/prueba_carga.py --url http://127.0.0.1:8000

Se usa 127.0.0.1 y no "localhost": en Windows, "localhost" prueba primero IPv6 (::1) y cada conexión
nueva tarda ~300 ms en caer a IPv4. Eso es del cliente, no de la API.
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
    "¿Cómo impactó el incidente de pagos de octubre en la opinión de los clientes? Quiero saber qué pasó, "
    "cuánto cambiaron las reseñas y qué deberíamos hacer.",
    "Mandá un cupón del 10% a los clientes con cobro doble por el incidente de pagos.",
    "¿Qué opinan los clientes desde el incidente de pagos? Compará las reseñas de antes y después.",
    "Compensá con un cupón del 5% a los clientes con el pedido demorado en el incidente de pagos.",
    "Mandá un cupón del 2% a los clientes con cobro doble.",
]
TERMINALES = {"DONE", "FAILED"}
REVISOR = "Roger (revisión de prueba)"


def decidir(pedido: str) -> dict:
    """Lo que haría el revisor con la acción que el agente propone para este pedido."""
    texto = pedido.lower()
    if "cupón" not in texto and "compens" not in texto:
        return {"aprobado": False, "revisor": REVISOR, "comentario": "Este pedido no pidió ninguna compensación."}
    if "demorado" in texto:
        return {"aprobado": False, "revisor": REVISOR, "comentario": "Muy caro para este segmento: no lo apruebo."}
    return {"aprobado": True, "revisor": REVISOR, "comentario": "Ok, es lo que dice el postmortem."}


def percentil(valores: list[float], p: float) -> float:
    """Percentil por rango más cercano (con 5 muestras, el p95 es el valor más alto)."""
    ordenados = sorted(valores)
    return ordenados[max(0, math.ceil(p / 100 * len(ordenados)) - 1)]


async def crear(cliente: httpx.AsyncClient, pedido: str) -> dict:
    inicio = time.perf_counter()
    respuesta = await cliente.post("/tasks", json={"pedido": pedido})
    respuesta.raise_for_status()
    return {**respuesta.json(), "pedido": pedido, "post_ms": round((time.perf_counter() - inicio) * 1000, 1),
            "t0": time.perf_counter()}


async def main(url: str, limite_s: float, salida: str) -> int:
    async with httpx.AsyncClient(base_url=url, timeout=30) as cliente:
        salud = (await cliente.get("/health")).json()
        print(f"API: {url} · modelos {salud['modelos']} · workers en la API: {salud['workers_en_api']}\n")

        jobs = await asyncio.gather(*(crear(cliente, p) for p in PEDIDOS))   # los 5 a la vez
        print(f"5 pedidos encolados. Respuesta del POST: {', '.join(f'{j['post_ms']} ms' for j in jobs)}\n")

        pendientes = {j["job_id"]: j for j in jobs}
        while pendientes and time.perf_counter() - jobs[0]["t0"] < limite_s:
            await asyncio.sleep(0.5)
            for job_id, job in list(pendientes.items()):
                estado = (await cliente.get(f"/tasks/{job_id}")).json()
                if estado["estado"] == "WAITING_APPROVAL":
                    decision = decidir(job["pedido"])
                    print(f"[{job_id[:8]}] pide aprobación: {estado['pregunta']['pregunta']}")
                    print(f"[{job_id[:8]}] -> {'APRUEBO' if decision['aprobado'] else 'RECHAZO'}: {decision.get('comentario', '')}")
                    job["espera_humana_s"] = round(time.perf_counter() - job["t0"], 2)
                    (await cliente.post(f"/tasks/{job_id}/approve", json=decision)).raise_for_status()
                elif estado["estado"] in TERMINALES:
                    job.update(final=estado, total_s=round(time.perf_counter() - job["t0"], 2))
                    pendientes.pop(job_id)

    print(f"\n{'job':<9} {'estado':<7} {'POST':>8} {'proceso':>8} {'total':>7}  pasos  pedido")
    for j in jobs:
        final = j.get("final", {})
        print(f"{j['job_id'][:8]:<9} {final.get('estado', 'TIMEOUT'):<7} {j['post_ms']:>6} ms {final.get('duracion_s', 0):>6} s "
              f"{j.get('total_s', 0):>5} s  {len(final.get('recorrido', [])):>5}  {j['pedido'][:55]}")
        if (final.get("resultado") or {}).get("resultado_accion"):
            print(f"{'':<9} acción: {final['resultado']['resultado_accion']}")
        if final.get("error"):
            print(f"{'':<9} error: {final['error']}")

    post = [j["post_ms"] for j in jobs]
    proceso = [j["final"]["duracion_s"] for j in jobs if "final" in j]
    print(f"\nRespuesta de la API (POST /tasks): p50 {percentil(post, 50)} ms · p95 {percentil(post, 95)} ms")
    if proceso:
        print(f"Tiempo de proceso del grafo:      p50 {percentil(proceso, 50)} s · p95 {percentil(proceso, 95)} s")
    print("Costo por ejecución y latencia por traza: en Phoenix (http://localhost:6006).")

    destino = Path(__file__).resolve().parent.parent / "resultados" / salida
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps({"fecha": datetime.now().isoformat(timespec="seconds"), "url": url, "modelos": salud["modelos"],
                                   "jobs": [{k: v for k, v in j.items() if k != "t0"} for j in jobs]},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detalle guardado en {destino.relative_to(destino.parent.parent)}")
    return 0 if not pendientes and all(j["final"]["estado"] == "DONE" for j in jobs) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    argumentos = argparse.ArgumentParser(description="5 pedidos concurrentes contra la API")
    argumentos.add_argument("--url", default="http://127.0.0.1:8000")
    argumentos.add_argument("--limite", type=float, default=600, help="segundos máximos de espera")
    argumentos.add_argument("--salida", default="prueba_carga.json", help="archivo de resultados/ donde se guarda la corrida")
    a = argumentos.parse_args()
    raise SystemExit(asyncio.run(main(a.url, a.limite, a.salida)))
