#!/bin/bash
# Проверяем FastAPI-сервис снаружи контейнера.
# Запускать с хоста: bash scripts/test_service.sh

SERVICE_URL="http://localhost:8000"

echo "Тестируем LLM-сервис (FastAPI)"

echo ""
echo "1) Healthcheck:"
curl -s "${SERVICE_URL}/health" | python3 -m json.tool

echo ""
echo "2) Запрос к модели через FastAPI:"
curl -s -X POST "${SERVICE_URL}/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Is this SMS spam or not? Reply only yes or no.\n\nSMS: WINNER!! As a valued network customer you have been selected to receive a 900 prize reward!",
        "system": "You are a spam detector. Answer only yes or no.",
        "temperature": 0.1
    }' | python3 -m json.tool

echo ""
echo "Готово"
