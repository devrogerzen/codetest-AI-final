"""Normalización de texto que comparten el RAG (BM25, embeddings locales) y las herramientas de reseñas."""
import re
import unicodedata

PALABRAS_VACIAS = {
    "a", "al", "algo", "como", "con", "cual", "cuales", "cuando", "de", "del", "el", "ella", "en", "entre",
    "es", "esa", "ese", "eso", "esta", "este", "esto", "hay", "la", "las", "le", "les", "lo", "los", "mas",
    "me", "mi", "no", "o", "para", "pero", "por", "que", "se", "si", "sin", "sobre", "su", "sus", "te", "tu",
    "un", "una", "uno", "unos", "y", "ya", "u", "fue", "son", "ser", "hace", "hacer", "hago", "puedo",
}


def normalizar(texto: str) -> str:
    """Minúsculas y sin acentos: "Caída" y "caida" son la misma palabra."""
    sin_acentos = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in sin_acentos if not unicodedata.combining(c)).lower()


def tokenizar(texto: str) -> list[str]:
    """Tokens para BM25: sin palabras vacías y conservando los guiones bajos, para que términos como
    pagos_modo_cola sigan siendo una sola palabra."""
    return [t for t in re.findall(r"[a-z0-9_]+", normalizar(texto)) if t not in PALABRAS_VACIAS]


def slug(texto: str) -> str:
    """"Runbook: Redis caído" -> "runbook-redis-caido" (IDs legibles y solo ASCII)."""
    return re.sub(r"[^a-z0-9]+", "-", normalizar(texto)).strip("-")
