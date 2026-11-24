"""Imprime el diagrama Mermaid del grafo de agentes, generado por LangGraph a partir del grafo compilado.

    python scripts/diagrama_grafo.py > docs/grafo.mmd

Usa los modelos simulados y no abre Redis ni Chroma: solo se compila el grafo para dibujarlo.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.update(SIN_API="1", TRAZAS="0")

from app.graph import construir_grafo, crear_modelos  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    grafo = construir_grafo(crear_modelos(sin_api=True), checkpointer=None, redis=None, buscador=None)
    print(grafo.get_graph().draw_mermaid())
