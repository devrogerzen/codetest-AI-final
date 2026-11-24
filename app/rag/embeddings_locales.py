"""Embeddings locales y deterministas, para correr y probar el RAG sin key (SIN_API=1 y tests).

No entienden significado como un modelo real: cada palabra (y cada par de palabras seguidas) suma en una
posición fija del vector, elegida por un hash. Alcanza para probar la ingesta, la base vectorial, los
filtros y la fusión híbrida de punta a punta; las métricas reales salen de los embeddings de Gemini.
"""
import hashlib
import math

from langchain_core.embeddings import Embeddings

from app.texto import tokenizar


class EmbeddingsLocales(Embeddings):
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.model = f"hash-{dimension}"

    def _posicion(self, termino: str) -> int:
        return int(hashlib.md5(termino.encode()).hexdigest(), 16) % self.dimension

    def _vector(self, texto: str) -> list[float]:
        tokens = tokenizar(texto)
        vector = [0.0] * self.dimension
        for termino in tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]:
            vector[self._posicion(termino)] += 1.0
        norma = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norma for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
