#!/usr/bin/env bash
# Levanta Colibrí Intelligence con una sola instrucción:   ./run.sh
#
#   ./run.sh          construye las imágenes y levanta redis, chroma, phoenix, la ingesta, la API y el worker
#   ./run.sh test     corre la suite de tests (sin Docker ni key)
#   ./run.sh down     baja todo (los volúmenes quedan: la memoria y la base vectorial no se pierden)
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-up}" in
  test)
    python -m pytest
    exit
    ;;
  down)
    docker compose down
    exit
    ;;
esac

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Creé .env a partir de .env.example. Completá GOOGLE_API_KEY (o poné SIN_API=1 para probar sin key)"
  echo "y volvé a correr ./run.sh"
  exit 1
fi

docker compose up -d --build

echo -n "Esperando a la API"
for _ in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo
    curl -fsS http://127.0.0.1:8000/health
    echo
    echo "Listo. API y documentación: http://127.0.0.1:8000/docs · Trazas: http://127.0.0.1:6006"
    exit 0
  fi
  echo -n "."
  sleep 2
done
echo
echo "La API no respondió a tiempo. Mirá los logs con: docker compose logs ingesta api worker"
exit 1
