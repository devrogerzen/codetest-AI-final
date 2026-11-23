"""Configuración: rutas, Redis, observabilidad, límites del orquestador y creación de los LLM.

Todo sale de variables de entorno (.env), así la misma imagen corre en local o en Docker.
"""
import os
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

BASE = Path(__file__).resolve().parent.parent
CONOCIMIENTO = BASE / "data" / "conocimiento"
RESENAS = BASE / "data" / "resenas.json"
SEGMENTOS = BASE / "data" / "segmentos.json"


def env(nombre: str, default: str | None = None, requerida: bool = False) -> str | None:
    valor = os.getenv(nombre) or default
    if requerida and not valor:
        raise ValueError(f"Falta {nombre} en el .env")
    return valor


# --- Redis: cola de trabajos, estado de cada job y checkpoints de LangGraph ---
REDIS_URL = env("REDIS_URL", "redis://localhost:6379")
COLA = "colibri:cola"
PREFIJO_JOB = "colibri:job:"
ENVIOS = "colibri:envios"            # registro de las acciones ejecutadas (efecto "real" de la herramienta crítica)
WORKERS = int(env("WORKERS", "3"))   # workers dentro del proceso de la API; 0 = correrlos aparte (python -m app.worker)

# --- Observabilidad (Arize Phoenix vía OpenInference / OpenTelemetry) ---
TRAZAS_ACTIVAS = env("TRAZAS", "1") == "1"
PHOENIX_ENDPOINT = env("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
PROYECTO_TRAZAS = env("PHOENIX_PROJECT_NAME", "colibri-orquestador")
PROMPT_VERSION = "v1.1"              # viaja como metadata en cada traza: si cambia un prompt, se ve en Phoenix

# --- Orquestador (lo de la pre-entrega 6) ---
MAX_DELEGACIONES = int(env("MAX_DELEGACIONES", "8"))
RECURSION_LIMIT = 50
RECURSION_LIMIT_ESPECIALISTA = 12

# --- Human-in-the-loop: una acción con costo estimado mayor a esto necesita aprobación humana ---
UMBRAL_APROBACION = float(env("UMBRAL_APROBACION", "50000"))   # pesos

SIN_API = env("SIN_API", "0") == "1"   # modelos simulados, sin key
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


def crear_llm(rol: str) -> BaseChatModel:
    """Un LLM por rol. El supervisor puede usar un modelo más capaz (SUPERVISOR_MODEL)."""
    provider = env("LLM_PROVIDER", "openai").lower()
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
