FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TIKTOKEN_CACHE_DIR=/srv/.tiktoken

COPY requirements.txt .
# El tokenizador de tiktoken se descarga en la construcción: la imagen no depende de internet para fragmentar
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY app ./app
COPY data ./data
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
