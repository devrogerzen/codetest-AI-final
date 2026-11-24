"""Límite de embeddings por minuto, del lado del cliente.

El plan gratis de Gemini cuenta cada texto como una llamada (100 por minuto). Una ingesta o una evaluación
que manda 150 textos de golpe choca contra el 429. Este envoltorio espera lo justo antes de pasarse, igual
que el InMemoryRateLimiter que usan los LLM (ver config.limitador).
"""
import re
import threading
import time
from collections import deque

from langchain_core.embeddings import Embeddings


class EmbeddingsConLimite(Embeddings):
    def __init__(self, base: Embeddings, textos_por_minuto: int, lote: int = 20):
        self.base = base
        self.model = getattr(base, "model", type(base).__name__)
        self.limite = textos_por_minuto
        self.lote = lote
        self._enviados: deque[float] = deque()   # momento en que se mandó cada texto del último minuto
        self._candado = threading.Lock()

    def _esperar_turno(self, cantidad: int) -> None:
        with self._candado:
            while True:
                ahora = time.monotonic()
                while self._enviados and ahora - self._enviados[0] >= 60:
                    self._enviados.popleft()
                if len(self._enviados) + cantidad <= self.limite:
                    self._enviados.extend([ahora] * cantidad)
                    return
                time.sleep(60 - (ahora - self._enviados[0]) + 0.1)

    def _con_reintentos(self, llamada, cantidad: int, intentos: int = 5):
        """Espera su turno y llama. Si igual llega un 429 (otro proceso gastó la cuota del minuto), espera lo
        que pide el proveedor y reintenta: es un error pasajero."""
        for intento in range(1, intentos + 1):
            self._esperar_turno(cantidad)
            try:
                return llamada()
            except Exception as e:
                if intento == intentos or not ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                    raise
                pedido = re.search(r"retry in ([\d.]+)s", str(e))
                time.sleep(float(pedido[1]) + 1 if pedido else 20 * intento)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectores: list[list[float]] = []
        for i in range(0, len(texts), self.lote):
            parte = texts[i:i + self.lote]
            vectores += self._con_reintentos(lambda: self.base.embed_documents(parte), len(parte))
        return vectores

    def embed_query(self, text: str) -> list[float]:
        return self._con_reintentos(lambda: self.base.embed_query(text), 1)
