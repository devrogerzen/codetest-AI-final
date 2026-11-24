"""Reseñas de clientes de Colibrí y el análisis de sentimiento que usa el analista.

La documentación ya no vive acá: la busca el RAG (app/rag) en la base vectorial.
"""
import json
import re
from datetime import date
from functools import cache
from typing import Any

from app.config import RESENAS
from app.texto import normalizar


@cache
def cargar_resenas() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(RESENAS.read_text(encoding="utf-8")))


def resenas_entre(desde: date, hasta: date, tema: str | None = None) -> list[dict[str, Any]]:
    tema_norm = normalizar(tema) if tema else None
    return [r for r in cargar_resenas()
            if desde <= date.fromisoformat(r["fecha"]) <= hasta
            and (tema_norm is None or tema_norm in normalizar(r["texto"]))]


def resenas_por_id(ids: list[int]) -> list[dict[str, Any]]:
    por_id = {r["id"]: r for r in cargar_resenas()}
    faltantes = [i for i in ids if i not in por_id]
    if faltantes:
        raise ValueError(f"No existen reseñas con id {faltantes}")
    return [por_id[i] for i in ids]


# Léxico de sentimiento en español (normalizado, sin acentos). "no", "sin" y "nunca" invierten
# la palabra que sigue: "sin problema" suma, "no confío" resta.
POSITIVAS = {"excelente", "rapido", "rapida", "impecable", "perfecto", "recomiendo", "buena", "bueno",
             "buenos", "bien", "contento", "contenta", "encanto", "facil", "comoda", "comodo", "seguro",
             "resolvieron", "resuelto", "volveria", "confio", "disculpas"}
NEGATIVAS = {"cobraron", "doble", "pesimo", "mal", "mala", "malo", "horrible", "demora", "tardo",
             "lento", "reclamo", "reclamar", "problema", "error", "desconfio", "molesta", "perder",
             "procesando", "nadie"}
NEGADORES = {"no", "sin", "nunca"}


def sentimiento(texto: str) -> tuple[str, float]:
    """Devuelve (etiqueta, puntaje entre -1 y 1)."""
    palabras = re.findall(r"[a-z]+", normalizar(texto))
    positivo = negativo = 0
    for i, palabra in enumerate(palabras):
        signo = 1 if palabra in POSITIVAS else -1 if palabra in NEGATIVAS else 0
        if signo and i > 0 and palabras[i - 1] in NEGADORES:
            signo = -signo
        positivo += signo > 0
        negativo += signo < 0
    puntaje = (positivo - negativo) / max(1, positivo + negativo)
    etiqueta = "positivo" if puntaje > 0.2 else "negativo" if puntaje < -0.2 else "neutro"
    return etiqueta, round(puntaje, 2)
