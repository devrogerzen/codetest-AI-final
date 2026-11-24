"""Fixtures de la suite: todo corre en memoria, sin Docker, sin Redis real y sin API key.

- Redis falso (fakeredis) para la cola, los jobs y las conversaciones.
- MemorySaver como checkpointer, pero con el MISMO serializador que usa Redis (JSON + lista de clases
  permitidas): un estado que no sobreviviría a Redis también falla acá.
- ChromaDB efímero (en memoria) con embeddings locales, cargado con la base de conocimiento real.
- Modelos simulados (app/modelo_simulado.py) en lugar de los LLM.
"""
import os
from contextlib import asynccontextmanager

os.environ.update(SIN_API="1", TRAZAS="0", EMBEDDINGS_PROVIDER="local", API_KEY="")   # antes de importar la app

import chromadb  # noqa: E402
import fakeredis  # noqa: E402
import httpx  # noqa: E402
import pytest  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer  # noqa: E402

from app.graph import construir_grafo, crear_modelos  # noqa: E402
from app.jobs import RepositorioJobs  # noqa: E402
from app.main import crear_app  # noqa: E402
from app.modelo_simulado import ModeloSimulado  # noqa: E402
from app.rag.almacen import abrir_almacen  # noqa: E402
from app.rag.buscador import BuscadorHibrido  # noqa: E402
from app.rag.embeddings_locales import EmbeddingsLocales  # noqa: E402
from app.rag.ingesta import ingestar  # noqa: E402
from app.state import PERMITIDOS_JSON  # noqa: E402
from app.worker import Recursos  # noqa: E402


@pytest.fixture(scope="session")
def cliente_chroma() -> chromadb.ClientAPI:
    return chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))


@pytest.fixture(scope="session")
def buscador(cliente_chroma) -> BuscadorHibrido:
    embeddings = EmbeddingsLocales()
    ingestar(embeddings=embeddings, cliente=cliente_chroma)
    return BuscadorHibrido(abrir_almacen(embeddings, cliente_chroma))


@pytest.fixture
def redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def serde() -> JsonPlusRedisSerializer:
    return JsonPlusRedisSerializer(allowed_json_modules=PERMITIDOS_JSON)


@pytest.fixture
def memoria(serde) -> MemorySaver:
    return MemorySaver(serde=serde)


@pytest.fixture
def abrir_api(redis, memoria, buscador):
    """abrir_api(modelos=..., workers=...) levanta la app completa (lifespan + workers) y devuelve un cliente
    HTTP. Abrirla dos veces con el mismo redis y la misma memoria simula un reinicio del servidor."""
    @asynccontextmanager
    async def abrir(modelos: dict | None = None, workers: int = 3):
        @asynccontextmanager
        async def recursos():
            grafo = construir_grafo(modelos or crear_modelos(sin_api=True), memoria, redis, buscador)
            yield Recursos(grafo, RepositorioJobs(redis), redis, buscador, ModeloSimulado(rol="asistente"))

        app = crear_app(recursos, workers=workers)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as cliente:
                yield cliente
    return abrir
