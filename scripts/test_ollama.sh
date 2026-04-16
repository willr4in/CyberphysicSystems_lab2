#!/bin/bash
# Проверяем, что Ollama работает прямо внутри контейнера.
# Запускать так: docker exec sms-spam-llm bash /opt/app/scripts/test_ollama.sh
# Или просто: bash scripts/test_ollama.sh (он сам зайдёт в контейнер)

echo "Проверка Ollama внутри контейнера"

# Спрашиваем у Ollama список моделей
echo ""
echo "1) Список загруженных моделей:"
docker exec sms-spam-llm curl -s http://localhost:11434/api/tags | python3 -m json.tool

echo ""
echo "2) Тестовый запрос к модели:"
docker exec sms-spam-llm curl -s http://localhost:11434/api/generate \
    -d '{
        "model": "qwen2.5:0.5b",
        "prompt": "Is this message spam? Answer yes or no: Congratulations! You won a free iPhone!",
        "stream": false
    }' | python3 -m json.tool

echo ""
echo "Готово"
