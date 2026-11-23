"""Prueba de reinicio: el estado de un job pausado sobrevive a reiniciar la API y los workers.

Es el desafío de la unidad 2 del módulo 7: pausar en la aprobación humana, reiniciar todo y aprobar después.

    1. python scripts/prueba_reinicio.py crear     -> crea un job que pide aprobación y espera WAITING_APPROVAL
    2. reiniciar la API y los workers (Ctrl+C y volver a levantarlos)
    3. python scripts/prueba_reinicio.py aprobar   -> aprueba ese job y espera que termine en DONE

Si el paso 3 termina en DONE, el checkpoint de LangGraph estaba en Redis y no en la memoria del proceso.
"""
import json
import sys
import time
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8000"
ARCHIVO = Path(__file__).resolve().parent.parent / "resultados" / "prueba_reinicio.json"
PEDIDO = "Mandá un cupón del 10% a los clientes con cobro doble por el incidente de pagos."


def esperar(cliente: httpx.Client, job_id: str, estados: set[str], limite: float = 60) -> dict:
    inicio = time.perf_counter()
    while time.perf_counter() - inicio < limite:
        job = cliente.get(f"/tasks/{job_id}").json()
        if job["estado"] in estados:
            return job
        time.sleep(0.5)
    raise TimeoutError(f"El job {job_id} no llegó a {estados}")


def crear(cliente: httpx.Client) -> None:
    job_id = cliente.post("/tasks", json={"pedido": PEDIDO}).json()["job_id"]
    job = esperar(cliente, job_id, {"WAITING_APPROVAL", "DONE", "FAILED"})
    print(f"Job {job_id}: {job['estado']}")
    print(f"Pregunta: {job.get('pregunta', {}).get('pregunta')}")
    ARCHIVO.parent.mkdir(exist_ok=True)
    ARCHIVO.write_text(json.dumps({"job_id": job_id, "antes_del_reinicio": job}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Ahora reiniciá la API y los workers, y corré: python scripts/prueba_reinicio.py aprobar")


def aprobar(cliente: httpx.Client) -> None:
    datos = json.loads(ARCHIVO.read_text(encoding="utf-8"))
    job_id = datos["job_id"]
    print(f"Después del reinicio, el job {job_id} está en: {cliente.get(f'/tasks/{job_id}').json()['estado']}")
    cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "Roger", "comentario": "Aprobado después del reinicio"})
    job = esperar(cliente, job_id, {"DONE", "FAILED"})
    print(f"Resultado: {job['estado']} · {job['resultado']['resultado_accion'] if job['resultado'] else job['error']}")
    print(f"Recorrido: {' -> '.join(job['recorrido'])}")
    datos["despues_del_reinicio"] = job
    ARCHIVO.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    paso = sys.argv[1] if len(sys.argv) > 1 else ""
    with httpx.Client(base_url=URL, timeout=30) as cliente:
        {"crear": crear, "aprobar": aprobar}.get(paso, lambda _: print(__doc__))(cliente)
