"""Pedidos de ejemplo y funciones de cliente que comparten los tests."""
import asyncio
import time

import httpx

ANALISIS = "¿Cómo impactó el incidente de pagos de octubre en la opinión de los clientes? Quiero saber qué pasó."
CUPON_10 = "Mandá un cupón del 10% a los clientes con cobro doble por el incidente de pagos."
CUPON_5_DEMORADOS = "Compensá con un cupón del 5% a los clientes con el pedido demorado en el incidente de pagos."
CUPON_2 = "Mandá un cupón del 2% a los clientes con cobro doble."
SEGUIMIENTO = "¿Y qué dice la documentación sobre el incidente de pagos y sus acciones pendientes?"


async def esperar(cliente: httpx.AsyncClient, job_id: str, estados: set[str], limite: float = 20) -> dict:
    """Polling, como haría un cliente real: consulta el job hasta que llega a uno de los estados."""
    inicio = time.perf_counter()
    while time.perf_counter() - inicio < limite:
        job = (await cliente.get(f"/tasks/{job_id}")).json()
        if job["estado"] in estados:
            return job
        await asyncio.sleep(0.05)
    return job


async def crear(cliente: httpx.AsyncClient, pedido: str, conversacion_id: str | None = None) -> dict:
    cuerpo = {"pedido": pedido} | ({"conversacion_id": conversacion_id} if conversacion_id else {})
    return (await cliente.post("/tasks", json=cuerpo)).json()
