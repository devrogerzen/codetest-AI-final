"""Fuentes de datos: la documentación de Colibrí (la de las pre-entregas 3 y 4 más el postmortem
del incidente de pagos) y las reseñas de clientes.

La búsqueda es léxica (BM25), sin embeddings: es la "búsqueda simulada sobre la Vector DB de
pre-entregas anteriores" que permite la consigna, y corre sin API key.
"""
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from functools import cache
from typing import Any

from app.config import CONOCIMIENTO, RESENAS

PALABRAS_VACIAS = {
    "a", "al", "algo", "como", "con", "cual", "de", "del", "el", "en", "es", "esta", "este", "hay",
    "la", "las", "lo", "los", "mas", "me", "mi", "para", "pero", "por", "que", "qué", "se", "si",
    "sin", "su", "sus", "un", "una", "uno", "y", "ya", "o", "u", "le", "les", "fue", "son", "ser",
}


def normalizar(texto: str) -> str:
    sin_acentos = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in sin_acentos if not unicodedata.combining(c)).lower()


def tokenizar(texto: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", normalizar(texto)) if t not in PALABRAS_VACIAS]


@dataclass(frozen=True)
class Seccion:
    documento_id: str
    titulo: str
    contenido: str


@cache
def cargar_documentacion() -> tuple[Seccion, ...]:
    """Cada sección (##) de cada Markdown y cada entrada del FAQ es un documento."""
    secciones: list[Seccion] = []
    for archivo in sorted(CONOCIMIENTO.glob("*.md")):
        texto = archivo.read_text(encoding="utf-8")
        titulo_doc = texto.splitlines()[0].lstrip("# ").strip()
        for bloque in re.split(r"^## ", texto, flags=re.MULTILINE)[1:]:
            titulo, _, cuerpo = bloque.partition("\n")
            slug = re.sub(r"[^a-z0-9]+", "-", normalizar(titulo)).strip("-")
            secciones.append(Seccion(f"{archivo.name}#{slug}", f"{titulo_doc} > {titulo.strip()}", cuerpo.strip()))
    for entrada in json.loads((CONOCIMIENTO / "faq.json").read_text(encoding="utf-8")):
        secciones.append(Seccion(f"faq.json#{entrada['id']}", f"FAQ > {entrada['pregunta']}", entrada["respuesta"]))
    return tuple(secciones)


def buscar_secciones(consulta: str, k: int = 3) -> list[tuple[Seccion, float]]:
    """BM25 clásico (k1=1.5, b=0.75) sobre título + contenido."""
    secciones = cargar_documentacion()
    docs = [tokenizar(f"{s.titulo} {s.contenido}") for s in secciones]
    largo_medio = sum(map(len, docs)) / len(docs)
    df = Counter(t for d in docs for t in set(d))
    puntajes = []
    for seccion, tokens in zip(secciones, docs):
        frecuencias = Counter(tokens)
        puntaje = 0.0
        for termino in set(tokenizar(consulta)):
            if termino not in frecuencias:
                continue
            idf = math.log(1 + (len(docs) - df[termino] + 0.5) / (df[termino] + 0.5))
            tf = frecuencias[termino]
            puntaje += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(tokens) / largo_medio))
        if puntaje > 0:
            puntajes.append((seccion, round(puntaje, 3)))
    return sorted(puntajes, key=lambda par: par[1], reverse=True)[:k]


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
