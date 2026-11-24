"""Prueba de reinicio: un job pausado y la memoria de su conversación sobreviven a reiniciar la API y los workers.

    python scripts/prueba_reinicio.py            # hace todo: crear -> docker compose restart api worker -> aprobar -> seguir
    python scripts/prueba_reinicio.py crear      # 1. crea un job que pide aprobación y espera WAITING_APPROVAL
    (reiniciar la API y los workers a mano)      # 2.
    python scripts/prueba_reinicio.py aprobar    # 3. lo aprueba y espera DONE
    python scripts/prueba_reinicio.py seguir     # 4. manda un segundo pedido a la MISMA conversación

Si el paso 3 termina en DONE, el checkpoint de LangGraph estaba en Redis y no en la memoria del proceso. Si el
paso 4 entiende "eso que aprobaste", la memoria de la conversación también sobrevivió.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8000"
ARCHIVO = Path(__file__).resolve().parent.parent / "resultados" / "prueba_reinicio.json"
PEDIDO = "Mandá un cupón del 10% a los clientes con cobro doble por el incidente de pagos."
SEGUIMIENTO = "¿Qué acción se tomó con los clientes con cobro doble y cuánto costó?"


def esperar(cliente: httpx.Client, job_id: str, estados: set[str], limite: float = 600) -> dict:
    inicio = time.perf_counter()
    while time.perf_counter() - inicio < limite:
        try:
            job = cliente.get(f"/tasks/{job_id}").json()
        except httpx.HTTPError:   # la API se está reiniciando
            time.sleep(1)
            continue
        if job["estado"] in estados:
            return job
        time.sleep(1)
    raise TimeoutError(f"El job {job_id} no llegó a {estados}")


def guardar(datos: dict) -> None:
    ARCHIVO.parent.mkdir(exist_ok=True)
    ARCHIVO.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def crear(cliente: httpx.Client) -> None:
    creado = cliente.post("/tasks", json={"pedido": PEDIDO}).json()
    job = esperar(cliente, creado["job_id"], {"WAITING_APPROVAL", "DONE", "FAILED"})
    print(f"Job {creado['job_id']} (conversación {creado['conversacion_id']}): {job['estado']}")
    print(f"Pregunta: {(job.get('pregunta') or {}).get('pregunta')}")
    guardar({"job_id": creado["job_id"], "conversacion_id": creado["conversacion_id"], "antes_del_reinicio": job})


def reiniciar() -> None:
    print("Reiniciando la API y los workers (docker compose restart api worker)...")
    subprocess.run(["docker", "compose", "restart", "api", "worker"], check=True, capture_output=True)
    with httpx.Client(base_url=URL, timeout=5) as cliente:
        for _ in range(90):
            try:
                if cliente.get("/health").status_code == 200:
                    print("La API volvió.")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
    raise TimeoutError("La API no volvió después del reinicio")


def aprobar(cliente: httpx.Client) -> None:
    datos = json.loads(ARCHIVO.read_text(encoding="utf-8"))
    job_id = datos["job_id"]
    print(f"Después del reinicio, el job está en: {cliente.get(f'/tasks/{job_id}').json()['estado']}")
    cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "Roger", "comentario": "Aprobado después del reinicio"})
    job = esperar(cliente, job_id, {"DONE", "FAILED"})
    print(f"Resultado: {job['estado']} · {job['resultado']['resultado_accion'] if job['resultado'] else job['error']}")
    datos["despues_del_reinicio"] = job
    guardar(datos)


def seguir(cliente: httpx.Client) -> None:
    datos = json.loads(ARCHIVO.read_text(encoding="utf-8"))
    creado = cliente.post("/tasks", json={"pedido": SEGUIMIENTO, "conversacion_id": datos["conversacion_id"]}).json()
    job = esperar(cliente, creado["job_id"], {"DONE", "FAILED", "WAITING_APPROVAL"})
    turnos = cliente.get(f"/conversations/{datos['conversacion_id']}").json()["turnos"]
    print(f"Segundo turno: {job['estado']} · turno {((job.get('resultado') or {}).get('turno'))} de la conversación")
    if job.get("resultado") and job["resultado"]["informe"]:
        print(f"Respuesta: {job['resultado']['informe']['resumen']}")
    print(f"La conversación tiene {len(turnos)} turnos en el checkpointer")
    datos.update(segundo_turno=job, turnos=turnos)
    guardar(datos)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    paso = sys.argv[1] if len(sys.argv) > 1 else "todo"
    with httpx.Client(base_url=URL, timeout=60) as cliente:
        if paso == "todo":
            crear(cliente)
            reiniciar()
            aprobar(cliente)
            seguir(cliente)
        else:
            {"crear": crear, "aprobar": aprobar, "seguir": seguir}.get(paso, lambda _: print(__doc__))(cliente)
