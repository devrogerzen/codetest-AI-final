"""RAG: fragmentación, ingesta incremental, base vectorial, búsqueda híbrida, defensas y métricas."""
import chromadb
import pytest
import tiktoken
from chromadb.config import Settings

from app.rag.almacen import EmbeddingsIncompatibles, abrir_almacen
from app.rag.documentos import cargar_documentos, fragmentar
from app.rag.embeddings_locales import EmbeddingsLocales
from app.rag.evaluacion import (cargar_golden_set, evaluar, precision_at_k, recall_at_k, reciprocal_rank,
                                relevantes)
from app.rag.ingesta import ingestar
from app.rag.respuestas import NO_LO_SE, RespuestaLLM, construir_cadena, verificar
from app.modelo_simulado import ModeloSimulado


@pytest.fixture
def cliente_nuevo(tmp_path):
    """Un Chroma vacío en una carpeta temporal, solo para este test (no toca la base de los demás tests)."""
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False))


def test_fragmentos_con_ids_deterministas_y_metadatos():
    primera, segunda = fragmentar(cargar_documentos()), fragmentar(cargar_documentos())
    assert [f.id for f in primera] == [f.id for f in segunda]       # mismo texto -> mismo ID
    assert len({f.id for f in primera}) == len(primera)              # sin choques
    runbook = next(f for f in primera if f.metadata["documento_id"] == "guia_incidentes.md#runbook-errores-5xx-en-pagos")
    assert runbook.page_content.startswith("Guía de incidentes de Colibrí > Runbook: errores 5xx en pagos")
    assert runbook.metadata["categoria"] == "operaciones"
    assert {"documento_id", "fuente", "titulo", "seccion", "categoria", "chunk_id", "tipo"} <= runbook.metadata.keys()


def test_chunking_por_tokens_respeta_el_tamano():
    contador = tiktoken.get_encoding("cl100k_base")
    chicos = fragmentar(cargar_documentos(), chunk_tokens=60, overlap_tokens=10)
    assert len(chicos) > len(fragmentar(cargar_documentos()))        # cortar más chico da más fragmentos
    for f in chicos:
        cuerpo = f.page_content.split("\n\n", 1)[1]                  # sin el encabezado "Título > Sección"
        assert len(contador.encode(cuerpo)) <= 60


def test_la_ingesta_es_incremental(cliente_nuevo):
    embeddings = EmbeddingsLocales()
    primera = ingestar(embeddings=embeddings, cliente=cliente_nuevo)
    segunda = ingestar(embeddings=embeddings, cliente=cliente_nuevo)
    assert primera["subidos"] == primera["fragmentos"] > 0
    assert segunda["subidos"] == 0 and segunda["borrados"] == 0      # nada cambió: no se paga ningún embedding

    fragmentos = fragmentar(cargar_documentos())
    fragmentos[0] = fragmentos[0].model_copy(update={"id": "cambiado1234abcd", "page_content": "texto nuevo"})
    tercera = ingestar(embeddings=embeddings, cliente=cliente_nuevo, fragmentos=fragmentos)
    assert (tercera["subidos"], tercera["borrados"]) == (1, 1)       # solo el fragmento que cambió


def test_no_mezcla_embeddings_de_modelos_distintos(cliente_nuevo):
    ingestar(embeddings=EmbeddingsLocales(dimension=512), cliente=cliente_nuevo)
    with pytest.raises(EmbeddingsIncompatibles):
        abrir_almacen(EmbeddingsLocales(dimension=256), cliente_nuevo)


async def test_busqueda_hibrida_encuentra_el_termino_tecnico(buscador):
    docs = await buscador.buscar("¿Qué flag hay que activar cuando falla el proveedor de pagos?", k=4)
    assert docs[0].metadata["documento_id"] == "guia_incidentes.md#runbook-errores-5xx-en-pagos"
    assert "pagos_modo_cola" in docs[0].page_content
    assert len({d.metadata["chunk_id"] for d in docs}) == len(docs)   # RRF fusiona duplicados por chunk_id


async def test_filtro_por_categoria(buscador):
    docs = await buscador.buscar("¿Qué hago si alguien pega una contraseña en un chat?", k=4, categoria="seguridad")
    assert docs and all(d.metadata["categoria"] == "seguridad" for d in docs)
    with pytest.raises(ValueError):
        await buscador.buscar("cualquier cosa", categoria="no-existe", modo="bm25")


@pytest.mark.parametrize("modo", ["bm25", "vectorial", "hibrido"])
async def test_los_tres_modos_devuelven_k(buscador, modo):
    docs = await buscador.buscar("despliegue canary", k=3, modo=modo)
    assert len(docs) == 3


def test_metricas_del_golden_set():
    recuperados = ["a", "b", "c", "d"]
    assert recall_at_k(recuperados, {"b", "z"}, 4) == 0.5
    assert precision_at_k(recuperados, {"b", "c"}, 4) == 0.5
    assert reciprocal_rank(recuperados, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(recuperados, {"z"}) == 0
    assert relevantes({"documento_id_esperado": ["x", "y"]}) == {"x", "y"}


async def test_evaluacion_completa_con_el_golden_set(buscador):
    golden = cargar_golden_set()
    resultado = await evaluar(buscador, golden, "hibrido", k=4)
    assert len(golden) == 15 and len(resultado["detalle"]) == 15
    assert 0 <= resultado["precision"] <= resultado["recall"] <= 1
    assert resultado["mrr"] > 0


def test_defensas_contra_fuentes_inventadas(buscador):
    docs = buscador.fragmentos[:2]
    real = docs[0].metadata["chunk_id"]
    salida = RespuestaLLM(respuesta="algo", encontrada_en_contexto=True, fuentes=[real, "inventado.md#x::0"])
    respuesta = verificar("pregunta", docs, salida)
    assert [f.chunk_id for f in respuesta.fuentes] == [real] and respuesta.descartadas == ["inventado.md#x::0"]

    sin_fuentes = verificar("pregunta", docs, RespuestaLLM(respuesta="algo", encontrada_en_contexto=True,
                                                            fuentes=["inventado.md#x::0"]))
    assert sin_fuentes.respuesta == NO_LO_SE and not sin_fuentes.encontrada


async def test_cadena_rag_responde_con_fuente_o_no_lo_se(buscador):
    cadena = construir_cadena(buscador, ModeloSimulado(rol="asistente"))
    respuesta = await cadena.ainvoke({"pregunta": "¿Qué flag hay que activar cuando falla el proveedor de pagos?"})
    assert respuesta.encontrada and respuesta.fuentes[0].documento_id == "guia_incidentes.md#runbook-errores-5xx-en-pagos"
    fuera = await cadena.ainvoke({"pregunta": "¿Cuál es la capital de Australia?"})
    assert fuera.respuesta == NO_LO_SE and fuera.fuentes == []
