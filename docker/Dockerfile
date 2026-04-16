# Базовый образ убунту
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Ставим всё, что понадобится: Python 3.12, curl для healthcheck и прочее
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    zstd \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Рабочая директория внутри контейнера
WORKDIR /opt/app

# Сначала копируем зависимости, чтобы Docker кешировал этот слой
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Копируем код приложения
COPY app/ ./app/

# Скрипт, который последовательно поднимает ollama и FastAPI
COPY docker/entrypoint.sh .
RUN chmod +x entrypoint.sh

# FastAPI слушает на 8000, ollama на 11434
EXPOSE 8000 11434

ENTRYPOINT ["./entrypoint.sh"]
