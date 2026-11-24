"""Conversaciones: el thread_id del checkpointer es la conversación, así que la memoria sobrevive entre
pedidos y a un reinicio de la API."""
from app.graph import crear_modelos
from app.modelo_simulado import ModeloSimulado
from tests.ayudas import ANALISIS, CUPON_10, SEGUIMIENTO, crear, esperar


class SupervisorQueAnota(ModeloSimulado):
    """Guarda lo que ve el supervisor de la conversación previa en cada decisión."""
    vistos: list = []

    def _supervisor(self, estado: dict):
        SupervisorQueAnota.vistos.append(estado.get("conversacion_previa", []))
        return super()._supervisor(estado)


async def test_dos_turnos_comparten_memoria(abrir_api):
    SupervisorQueAnota.vistos = []
    modelos = crear_modelos(sin_api=True) | {"supervisor": SupervisorQueAnota(rol="supervisor")}
    async with abrir_api(modelos) as cliente:
        primero = await crear(cliente, ANALISIS)
        job1 = await esperar(cliente, primero["job_id"], {"DONE", "FAILED"})
        conversacion = primero["conversacion_id"]
        assert SupervisorQueAnota.vistos[0] == []   # el primer turno no tiene memoria

        segundo = await crear(cliente, SEGUIMIENTO, conversacion)
        job2 = await esperar(cliente, segundo["job_id"], {"DONE", "FAILED"})
        respuesta = (await cliente.get(f"/conversations/{conversacion}")).json()

    assert job1["estado"] == job2["estado"] == "DONE"
    assert segundo["conversacion_id"] == conversacion
    assert job2["resultado"]["turno"] == 2
    # En el segundo turno el supervisor ve el pedido y el resumen del primero
    previa = SupervisorQueAnota.vistos[-1]
    assert previa and previa[0]["pedido"] == ANALISIS
    assert previa[0]["resumen"] == job1["resultado"]["informe"]["resumen"]
    # Lo propio de cada turno arranca de cero: el registro de aportes vuelve a empezar en el paso 1
    assert job2["resultado"]["aportes"][0]["paso"] == 1
    assert [t["pedido"] for t in respuesta["turnos"]] == [ANALISIS, SEGUIMIENTO]
    assert [t["referencia"] for t in respuesta["turnos"]] == ["conversacion#turno-1", "conversacion#turno-2"]
    assert respuesta["jobs"] == [primero["job_id"], segundo["job_id"]]


async def test_un_pedido_a_la_vez_por_conversacion(abrir_api):
    async with abrir_api(workers=0) as cliente:   # sin workers: el primer job queda pendiente
        primero = await crear(cliente, ANALISIS)
        r = await cliente.post("/tasks", json={"pedido": SEGUIMIENTO, "conversacion_id": primero["conversacion_id"]})
        otra = await cliente.post("/tasks", json={"pedido": SEGUIMIENTO})   # otra conversación sí entra
    assert r.status_code == 409
    assert r.json()["job_activo"] == primero["job_id"]
    assert otra.status_code == 202


async def test_mientras_espera_aprobacion_la_conversacion_sigue_tomada(abrir_api):
    async with abrir_api() as cliente:
        pausado = await crear(cliente, CUPON_10)
        await esperar(cliente, pausado["job_id"], {"WAITING_APPROVAL"})
        r = await cliente.post("/tasks", json={"pedido": SEGUIMIENTO, "conversacion_id": pausado["conversacion_id"]})
        assert r.status_code == 409
        await cliente.post(f"/tasks/{pausado['job_id']}/approve", json={"aprobado": True, "revisor": "Ana"})
        await esperar(cliente, pausado["job_id"], {"DONE"})
        siguiente = await cliente.post("/tasks", json={"pedido": SEGUIMIENTO, "conversacion_id": pausado["conversacion_id"]})
    assert siguiente.status_code == 202   # terminado el turno, la conversación se libera


async def test_la_conversacion_sobrevive_a_un_reinicio(abrir_api):
    async with abrir_api() as cliente:
        primero = await crear(cliente, ANALISIS)
        await esperar(cliente, primero["job_id"], {"DONE"})
    # Otra instancia de la API con el mismo Redis y el mismo checkpointer
    async with abrir_api() as cliente:
        segundo = await crear(cliente, SEGUIMIENTO, primero["conversacion_id"])
        job2 = await esperar(cliente, segundo["job_id"], {"DONE", "FAILED"})
        turnos = (await cliente.get(f"/conversations/{primero['conversacion_id']}")).json()["turnos"]
    assert job2["estado"] == "DONE" and job2["resultado"]["turno"] == 2
    assert len(turnos) == 2


async def test_conversacion_inexistente_da_404(abrir_api):
    async with abrir_api(workers=0) as cliente:
        assert (await cliente.get("/conversations/no-existe-123")).status_code == 404
