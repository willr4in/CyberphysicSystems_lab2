"""
Тестовый скрипт — проверяем работу LLM-сервиса с хоста через requests.

Запуск: python scripts/test_service.py
Сервис должен быть запущен (docker compose up).
"""

import requests

SERVICE_URL = "http://localhost:8000"


def check_health():
    """Проверим, что сервис вообще отвечает."""
    resp = requests.get(f"{SERVICE_URL}/health", timeout=5)
    resp.raise_for_status()
    print(f"Healthcheck: {resp.json()}")
    return resp.json()["status"] == "ok"


def send_message(text: str, system: str = "", temperature: float = 0.1):
    """
    Отправляем сообщение модели и получаем ответ.

    Просто передаём текст SMS и системный промпт,
    а модель решает — спам или нет.
    """
    payload = {
        "prompt": text,
        "system": system,
        "temperature": temperature,
    }
    resp = requests.post(
        f"{SERVICE_URL}/api/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    """Запускаем пару тестовых запросов и смотрим, что модель отвечает."""
    print("Тест LLM-сервиса\n")

    if not check_health():
        print("Сервис не отвечает!")
        return

    # Тест 1 — очевидный спам
    print("\nТест 1: спам")
    result = send_message(
        text="Is this SMS spam? Answer yes or no.\n\nSMS: WINNER!! You have been selected to receive a 900 prize reward! Call 09061701461.",
        system="You are a spam detector. Answer only yes or no.",
    )
    print(f"Ответ модели: {result['response']}")

    # Тест 2 — обычное сообщение
    print("\nТест 2: не спам")
    result = send_message(
        text="Is this SMS spam? Answer yes or no.\n\nSMS: Hey, are you coming to the party tonight? Let me know!",
        system="You are a spam detector. Answer only yes or no.",
    )
    print(f"Ответ модели: {result['response']}")

    print("\nТесты завершены")


if __name__ == "__main__":
    main()
