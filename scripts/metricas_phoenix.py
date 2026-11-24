"""Reporte de costo y latencia por ejecución, leído de Arize Phoenix (su API GraphQL).

Phoenix calcula el costo de cada span de LLM con los tokens y el precio del modelo; este script lo junta
por traza (una traza = un pedido) y por tipo de pedido, y calcula los percentiles de latencia.

    python scripts/metricas_phoenix.py                  # imprime el reporte y lo guarda en resultados/
    python scripts/metricas_phoenix.py --url http://127.0.0.1:6006 --proyecto colibri-intelligence
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx

CONSULTA = """
query ($id: ID!, $despues: String) {
  node(id: $id) {
    ... on Project {
      name traceCount sessionCount tokenCountTotal tokenCountPrompt tokenCountCompletion
      costSummary { total { cost } prompt { cost } completion { cost } }
      p50: latencyMsQuantile(probability: 0.5)
      p95: latencyMsQuantile(probability: 0.95)
      spans(first: 100, after: $despues, filterCondition: "parent_id is None") {
        pageInfo { hasNextPage endCursor }
        edges { node { name latencyMs startTime metadata
          trace { traceId numSpans costSummary { total { cost tokens } } } } }
      }
    }
  }
}"""


def percentil(valores: list[float], p: float) -> float:
    ordenados = sorted(valores)
    return ordenados[max(0, math.ceil(p / 100 * len(ordenados)) - 1)] if ordenados else 0.0


def main(url: str, proyecto: str, salida: str) -> None:
    with httpx.Client(base_url=url, timeout=60) as cliente:
        def gql(consulta: str, variables: dict | None = None) -> dict:
            respuesta = cliente.post("/graphql", json={"query": consulta, "variables": variables or {}}).json()
            if "errors" in respuesta:
                raise RuntimeError(respuesta["errors"][0]["message"])
            return respuesta["data"]

        proyectos = gql("{ projects { edges { node { id name } } } }")["projects"]["edges"]
        id_proyecto = next(p["node"]["id"] for p in proyectos if p["node"]["name"] == proyecto)
        trazas, despues, resumen = [], None, None
        while True:
            datos = gql(CONSULTA, {"id": id_proyecto, "despues": despues})["node"]
            resumen = resumen or datos
            for borde in datos["spans"]["edges"]:
                span = borde["node"]
                meta = json.loads(span["metadata"]) if isinstance(span["metadata"], str) else (span["metadata"] or {})
                costo = span["trace"]["costSummary"]["total"]
                trazas.append({"traza": span["trace"]["traceId"], "inicio": span["startTime"], "raiz": span["name"],
                               "tipo": meta.get("tipo", "?"), "job_id": meta.get("job_id"),
                               "latencia_s": round(span["latencyMs"] / 1000, 2), "spans": span["trace"]["numSpans"],
                               "tokens": costo["tokens"], "costo_usd": costo["cost"]})
            if not datos["spans"]["pageInfo"]["hasNextPage"]:
                break
            despues = datos["spans"]["pageInfo"]["endCursor"]

    por_tipo: dict[str, list[dict]] = defaultdict(list)
    for t in trazas:
        por_tipo[t["tipo"]].append(t)
    costo_total = resumen["costSummary"]["total"]["cost"] or 0
    print(f"Proyecto {proyecto}: {resumen['traceCount']} trazas · {resumen['sessionCount']} sesiones · "
          f"{resumen['tokenCountTotal']:,} tokens · US${costo_total:.4f}")
    print(f"Latencia de las trazas: p50 {resumen['p50'] / 1000:.1f} s · p95 {resumen['p95'] / 1000:.1f} s\n")
    print(f"{'tipo':<12} {'trazas':>6} {'costo medio':>12} {'costo máx':>10} {'tokens medio':>13} {'p50':>7} {'p95':>7}")
    for tipo, lista in sorted(por_tipo.items()):
        costos = [t["costo_usd"] or 0 for t in lista]
        latencias = [t["latencia_s"] for t in lista]
        print(f"{tipo:<12} {len(lista):>6} {sum(costos) / len(costos):>12.5f} {max(costos):>10.5f} "
              f"{sum(t['tokens'] or 0 for t in lista) / len(lista):>13,.0f} {percentil(latencias, 50):>6.1f}s "
              f"{percentil(latencias, 95):>6.1f}s")
    print("\n(iniciar = un pedido al equipo de agentes hasta que termina o se pausa; reanudar = lo que corre "
          "después de la aprobación humana; ask/ask_stream = RAG directo)")

    # Costo por ejecución de punta a punta: un pedido con aprobación humana son dos trazas (iniciar + reanudar)
    por_job: dict[str, dict] = {}
    for t in trazas:
        if t["job_id"]:
            job = por_job.setdefault(t["job_id"], {"job_id": t["job_id"], "trazas": 0, "costo_usd": 0.0, "tokens": 0,
                                                   "latencia_s": 0.0, "aprobacion_humana": False})
            job["trazas"] += 1
            job["costo_usd"] += t["costo_usd"] or 0
            job["tokens"] += t["tokens"] or 0
            job["latencia_s"] = round(job["latencia_s"] + t["latencia_s"], 2)
            job["aprobacion_humana"] |= t["tipo"] == "reanudar"
    jobs = list(por_job.values())
    if jobs:
        costos = [j["costo_usd"] for j in jobs]
        print(f"\nCosto por pedido al equipo de agentes ({len(jobs)} pedidos): medio US${sum(costos) / len(costos):.5f} · "
              f"mín US${min(costos):.5f} · máx US${max(costos):.5f} · p95 de latencia de proceso "
              f"{percentil([j['latencia_s'] for j in jobs], 95):.1f} s")

    destino = Path(__file__).resolve().parent.parent / "resultados" / salida
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps({
        "fecha": datetime.now().isoformat(timespec="seconds"), "proyecto": proyecto,
        "trazas": resumen["traceCount"], "sesiones": resumen["sessionCount"],
        "tokens": {"total": resumen["tokenCountTotal"], "prompt": resumen["tokenCountPrompt"],
                   "completion": resumen["tokenCountCompletion"]},
        "costo_usd": {"total": costo_total, "prompt": resumen["costSummary"]["prompt"]["cost"],
                      "completion": resumen["costSummary"]["completion"]["cost"]},
        "latencia_s": {"p50": round(resumen["p50"] / 1000, 2), "p95": round(resumen["p95"] / 1000, 2)},
        "por_pedido": jobs,
        "por_traza": sorted(trazas, key=lambda t: t["inicio"]),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado en resultados/{salida}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    argumentos = argparse.ArgumentParser(description="Costo y latencia por ejecución, desde Phoenix")
    argumentos.add_argument("--url", default="http://127.0.0.1:6006")
    argumentos.add_argument("--proyecto", default="colibri-intelligence")
    argumentos.add_argument("--salida", default="metricas_phoenix.json")
    a = argumentos.parse_args()
    main(a.url, a.proyecto, a.salida)
