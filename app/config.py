"""Configuración: rutas, Redis, RAG, observabilidad, límites del orquestador y creación de LLM y embeddings.

Todo sale de variables de entorno (.env): ningún modelo, colección ni URL está fijo en el código, así la
misma imagen corre en local, en Docker o con otro proveedor sin tocar nada.
"""
import os
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

load_dotenv()

BASE = Path(__file__).resolve().parent.parent
CONOCIMIENTO = BASE / "data" / "conocimiento"
RESENAS = BASE / "data" / "resenas.json"
SEGMENTOS = BASE / "data" / "segmentos.json"
GOLDEN_SET = BASE / "data" / "golden_set.json"


def env(nombre: str, default: str | None = None, requerida: bool = False) -> str | None:
    valor = os.getenv(nombre) or default
    if requerida and not valor:
        raise ValueError(f"Falta {nombre} en el .env")
    return valor


SIN_API = env("SIN_API", "0") == "1"   # modelos simulados y embeddings locales: sin key y sin costo

# --- Redis: cola de trabajos, estado de cada job, conversaciones y checkpoints de LangGraph ---
REDIS_URL = env("REDIS_URL", "redis://localhost:6379")
COLA = "colibri:cola"
PREFIJO_JOB = "colibri:job:"
PREFIJO_CONVERSACION = "colibri:conversacion:"
ENVIOS = "colibri:envios"            # registro de las acciones ejecutadas (efecto "real" de la herramienta crítica)
WORKERS = int(env("WORKERS", "3"))   # workers dentro del proceso de la API; 0 = correrlos aparte (python -m app.worker)
API_KEY = env("API_KEY")             # si está definida, la API exige el header X-API-Key

# --- RAG: base de conocimiento, fragmentos y base vectorial (ChromaDB) ---
# Con CHROMA_HOST usa el servidor de Chroma (el de docker-compose); sin él, una carpeta local persistente.
CHROMA_HOST = env("CHROMA_HOST")
CHROMA_PORT = int(env("CHROMA_PORT", "8000"))
CHROMA_DIR = Path(env("CHROMA_DIR", str(BASE / "chroma_db")))
CHUNK_TOKENS = int(env("CHUNK_TOKENS", "400"))
OVERLAP_TOKENS = int(env("OVERLAP_TOKENS", "40"))
TOP_K = int(env("TOP_K", "4"))                  # fragmentos que devuelve el buscador (y k de las métricas)
PESO_BM25 = float(env("PESO_BM25", "0.5"))     # peso de BM25 en la fusión RRF; los vectores pesan 1 - PESO_BM25

# --- Observabilidad (Arize Phoenix vía OpenInference / OpenTelemetry) ---
TRAZAS_ACTIVAS = env("TRAZAS", "1") == "1"
PHOENIX_ENDPOINT = env("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
PROYECTO_TRAZAS = env("PHOENIX_PROJECT_NAME", "colibri-intelligence")
PROMPT_VERSION = "v2.0"              # viaja como metadata en cada traza: si cambia un prompt, se ve en Phoenix

# --- Orquestador ---
MAX_DELEGACIONES = int(env("MAX_DELEGACIONES", "8"))
RECURSION_LIMIT = 50
RECURSION_LIMIT_ESPECIALISTA = 12
TURNOS_DE_CONTEXTO = 3               # turnos anteriores de la conversación que ve el supervisor

# --- Human-in-the-loop: una acción con costo estimado mayor a esto necesita aprobación humana ---
UMBRAL_APROBACION = float(env("UMBRAL_APROBACION", "50000"))   # pesos

LLM_MAX_RETRIES = int(env("LLM_MAX_RETRIES", "2"))   # reintentos ante 429/5xx (en planes gratis conviene subirlo)
LLM_RPM = float(env("LLM_RPM", "0"))                 # tope de llamadas por minuto a cada modelo (0 = sin tope)


def resumir_error(e: Exception) -> str:
    lineas = [linea.strip() for linea in str(e).splitlines() if linea.strip()][:2]
    return f"{type(e).__name__}: {' '.join(lineas)[:250]}"


@cache
def limitador(modelo: str):
    """Un limitador por modelo, compartido por todos los agentes del proceso que lo usan.

    Los planes gratis cortan por llamadas por minuto y por modelo (Gemini: 15). Mejor esperar del lado
    del cliente que chocar contra el 429 y depender de los reintentos.
    """
    if not LLM_RPM:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter
    return InMemoryRateLimiter(requests_per_second=LLM_RPM / 60, check_every_n_seconds=0.1, max_bucket_size=1)


def proveedor_llm() -> str:
    return env("LLM_PROVIDER", "openai").lower()


def crear_llm(rol: str) -> BaseChatModel:
    """Un LLM por rol (Factory). El supervisor puede usar un modelo más capaz (SUPERVISOR_MODEL)."""
    provider = proveedor_llm()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        env("OPENAI_API_KEY", requerida=True)
        modelo = env("OPENAI_MODEL", "gpt-4o-mini")
        if rol == "supervisor":
            modelo = env("SUPERVISOR_MODEL", modelo)
        return ChatOpenAI(model=modelo, temperature=0, base_url=env("OPENAI_BASE_URL"), timeout=60,
                          max_retries=LLM_MAX_RETRIES, rate_limiter=limitador(modelo))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        env("ANTHROPIC_API_KEY", requerida=True)
        modelo = env("ANTHROPIC_MODEL", "claude-haiku-4-5")
        if rol == "supervisor":
            modelo = env("SUPERVISOR_MODEL", modelo)
        # Sin temperature: el SDK actual de Anthropic ya no la acepta
        return ChatAnthropic(model=modelo, max_tokens=2048, timeout=60, max_retries=LLM_MAX_RETRIES,
                             rate_limiter=limitador(modelo))
    if provider == "google":
        # Integración nativa: Gemini 3 exige devolver la "thought signature" de cada llamada a herramienta,
        # y el cliente de OpenAI (vía su endpoint compatible) la pierde en el segundo turno del agente.
        from langchain_google_genai import ChatGoogleGenerativeAI
        env("GOOGLE_API_KEY", requerida=True)
        modelo = env("GOOGLE_MODEL", "gemini-3.1-flash-lite")
        if rol == "supervisor":
            modelo = env("SUPERVISOR_MODEL", modelo)
        # Sin temperature: en Gemini 3 Google recomienda dejar la que trae por defecto
        return ChatGoogleGenerativeAI(model=modelo, timeout=60, max_retries=LLM_MAX_RETRIES,
                                      rate_limiter=limitador(modelo))
    raise ValueError(f"LLM_PROVIDER no soportado: {provider!r} (usá openai, anthropic o google)")


def nombre_modelo(modelo: BaseChatModel) -> str:
    return getattr(modelo, "model_name", None) or getattr(modelo, "model", None) or modelo._llm_type


def proveedor_embeddings() -> str:
    """local (sin key), google u openai. Anthropic no tiene embeddings: con LLM_PROVIDER=anthropic hay que elegir otro."""
    return env("EMBEDDINGS_PROVIDER", "local" if SIN_API else proveedor_llm()).lower()


def crear_embeddings() -> Embeddings:
    proveedor = proveedor_embeddings()
    if proveedor == "local":
        from app.rag.embeddings_locales import EmbeddingsLocales
        return EmbeddingsLocales()
    if proveedor == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        env("GOOGLE_API_KEY", requerida=True)
        return GoogleGenerativeAIEmbeddings(model=env("EMBEDDINGS_MODEL", "gemini-embedding-001"))
    if proveedor == "openai":
        from langchain_openai import OpenAIEmbeddings
        env("OPENAI_API_KEY", requerida=True)
        return OpenAIEmbeddings(model=env("EMBEDDINGS_MODEL", "text-embedding-3-small"))
    raise ValueError(f"EMBEDDINGS_PROVIDER no soportado: {proveedor!r} (usá local, google u openai)")


def nombre_embeddings(embeddings: Embeddings) -> str:
    modelo = getattr(embeddings, "model", None) or type(embeddings).__name__
    return f"{proveedor_embeddings()}/{str(modelo).removeprefix('models/')}"


def nombre_coleccion() -> str:
    """Una colección por familia de embeddings: los vectores de dos modelos distintos no se pueden mezclar."""
    return env("CHROMA_COLLECTION", "colibri-conocimiento" + ("-local" if proveedor_embeddings() == "local" else ""))
