"""El validador determinista: el código, y no el LLM, decide si el informe sale."""
from app.agents.validator import problemas_del_informe
from app.state import Hallazgo, Informe, Metrica, Turno, estado_turno


def estado(**cambios) -> dict:
    base = estado_turno("¿Cómo impactó el incidente de pagos en las reseñas?") | {
        "hallazgos": [Hallazgo(tipo="documento", referencia="postmortem_pagos_2026-10.md#impacto-en-clientes",
                               contenido="312 clientes fueron cobrados dos veces.")],
        "metricas": [Metrica(nombre="promedio[antes]", valor=4.36), Metrica(nombre="promedio[después]", valor=2.44)],
        "aportes": [], "historial": [],
    }
    return base | cambios


def informe(**cambios) -> Informe:
    base = dict(titulo="Impacto", resumen="El promedio bajó de 4.36 a 2.44 y hubo 312 cobros dobles.",
                hallazgos=["312 clientes con cobro doble"], metricas_citadas=["promedio[antes]", "promedio[después]"],
                fuentes=["postmortem_pagos_2026-10.md#impacto-en-clientes"])
    return Informe(**(base | cambios))


def test_un_informe_correcto_pasa():
    assert problemas_del_informe(estado(informe=informe())) == []


def test_un_numero_inventado_se_rechaza():
    problemas = problemas_del_informe(estado(informe=informe(resumen="El promedio bajó a 1.90 y hubo 312 cobros dobles.")))
    assert any("1.90" in p for p in problemas)


def test_una_fuente_que_no_se_recupero_se_rechaza():
    problemas = problemas_del_informe(estado(informe=informe(fuentes=["inventado.md#algo"])))
    assert any("inventado.md#algo" in p for p in problemas)


def test_una_metrica_inexistente_se_rechaza():
    problemas = problemas_del_informe(estado(informe=informe(metricas_citadas=["promedio[otro]"])))
    assert any("promedio[otro]" in p for p in problemas)


def test_sin_hallazgos_el_informe_va_sin_fuentes():
    """Un pedido que solo pide una acción no investiga nada: el informe no cita fuentes, y si inventa una, se rechaza."""
    accion = "Se enviaron 312 cupones del 2% (costo estimado $42.432), aprobados automáticamente."
    sin_nada = estado(hallazgos=[], metricas=[], resultado_accion=accion)
    limpio = informe(resumen="Se enviaron 312 cupones del 2%.", hallazgos=["Compensación enviada"], metricas_citadas=[],
                     fuentes=[], accion=accion)
    assert problemas_del_informe(sin_nada | {"informe": limpio}) == []
    inventado = limpio.model_copy(update={"fuentes": ["Log de ejecución de compensación"]})
    assert any("Log de ejecución" in p for p in problemas_del_informe(sin_nada | {"informe": inventado}))


def test_un_turno_anterior_es_una_fuente_valida():
    """En el segundo turno no hace falta volver a investigar: el informe puede citar el turno anterior."""
    anterior = Turno(referencia="conversacion#turno-1", pedido="¿Qué pasó?", titulo="Incidente",
                     resumen="312 clientes con cobro doble.")
    accion = "Se enviaron 312 cupones del 10% (costo estimado $212.160)."
    segundo = estado(hallazgos=[], metricas=[], resultado_accion=accion, historial=[anterior])
    citando = informe(resumen="Se enviaron 312 cupones del 10%.", hallazgos=["Compensación enviada"],
                      metricas_citadas=[], fuentes=["conversacion#turno-1"], accion=accion)
    assert problemas_del_informe(segundo | {"informe": citando}) == []
    otro_nombre = citando.model_copy(update={"fuentes": ["Informe del incidente del 14/10"]})
    problemas = problemas_del_informe(segundo | {"informe": otro_nombre})
    assert problemas and "conversacion#turno-1" in problemas[0]   # el mensaje dice cuáles son las fuentes válidas


def test_los_numeros_de_un_turno_anterior_valen():
    anterior = Turno(pedido="p", titulo="t", resumen="Se compensó a 1.240 clientes con pedidos demorados.")
    texto = "El promedio bajó de 4.36 a 2.44; antes se habló de 1.240 pedidos demorados y 312 cobros dobles."
    assert problemas_del_informe(estado(informe=informe(resumen=texto), historial=[anterior])) == []
