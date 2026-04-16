#!/bin/bash
set -e

echo "Запускаем сервер Ollama"
ollama serve &

# Даём ollama время подняться — без этого pull может упасть
echo "Ждём, пока Ollama будет готов"
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama готов!"
        break
    fi
    echo "Попытка $i/30 — ждём ещё..."
    sleep 2
done

# Скачиваем модель (если уже скачана — пропустит)
echo "Загружаем модель Qwen2.5:0.5B"
ollama pull qwen2.5:0.5b

echo "Модель загружена, запускаем FastAPI"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
