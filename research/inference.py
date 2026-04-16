"""
Прогоняем датасет через LLM-сервис с каждой техникой промптинга.

Скрипт берёт SMS из датасета, отправляет в наш FastAPI-сервис
с соответствующим промптом и сохраняет результаты в CSV.

Запуск:
    python research/inference.py                    # все техники
    python research/inference.py --technique cot    # только CoT
    python research/inference.py --limit 100        # первые 100 сообщений
"""

import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import requests

# Добавляем корень проекта в путь, чтобы импортировать промпты
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import PROMPTS

SERVICE_URL = "http://localhost:8000"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_dataset(path: str) -> pd.DataFrame:
    """
    Загружаем датасет и приводим к нужному формату.

    Всё как в download_dataset.py — v1/v2 переименовываем,
    label маппим в числа.
    """
    df = pd.read_csv(path, encoding="latin-1")
    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "text"]
    df["target"] = (df["label"] == "spam").astype(int)
    return df


def query_llm(prompt: str, system: str, temperature: float = 0.1) -> str:
    """
    Отправляем запрос в наш LLM-сервис и получаем текст ответа.

    Если сервис не отвечает — возвращаем пустую строку,
    чтобы не обрушить весь прогон из-за одного таймаута.
    """
    try:
        resp = requests.post(
            f"{SERVICE_URL}/api/generate",
            json={
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except (requests.RequestException, ValueError) as e:
        print(f"  Ошибка запроса: {e}")
        return ""


def parse_zero_shot_response(response: str) -> int:
    """
    Парсим ответ zero-shot — модель должна была ответить spam или ham.

    Тут ищем ключевые слова. Если модель ответила что-то непонятное —
    возвращаем -1 (потом учтём как ошибку).
    """
    text = response.strip().lower()

    # Иногда модель пишет "spam" или "ham" с пояснениями вокруг
    if "spam" in text and "ham" not in text:
        return 1
    if "ham" in text and "spam" not in text:
        return 0

    # Бывает, модель пишет "this is spam" или "not spam"
    if "not spam" in text or "not a spam" in text:
        return 0
    if "spam" in text:
        return 1

    # Совсем непонятный ответ
    return -1


def parse_json_response(response: str) -> tuple[str, int]:
    """
    Парсим JSON-ответ от модели (для CoT, few-shot, CoT+few-shot).

    Модель должна вернуть {"reasoning": "...", "verdict": 0 или 1}.
    Но маленькие модели иногда добавляют текст вокруг JSON,
    поэтому пытаемся вытащить JSON из ответа.
    """
    json_match = re.search(r'\{[^{}]*"reasoning"[^{}]*"verdict"[^{}]*\}', response)
    if json_match:
        try:
            data = json.loads(json_match.group())
            verdict = int(data.get("verdict", -1))
            reasoning = data.get("reasoning", "")
            if verdict in (0, 1):
                return reasoning, verdict
        except (json.JSONDecodeError, ValueError):
            pass

    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            verdict = int(data.get("verdict", -1))
            reasoning = data.get("reasoning", "")
            if verdict in (0, 1):
                return reasoning, verdict
        except (json.JSONDecodeError, ValueError):
            pass

    verdict_match = re.search(r'"verdict"\s*:\s*([01])', response)
    if verdict_match:
        return "", int(verdict_match.group(1))

    return "", -1


def run_inference(df: pd.DataFrame, technique: str) -> pd.DataFrame:
    """
    Прогоняем все SMS через модель с указанной техникой.

    Для каждого сообщения формируем промпт по шаблону,
    отправляем в LLM-сервис, парсим ответ.
    Результаты складываем в DataFrame.
    """
    config = PROMPTS[technique]
    system_prompt = config["system"]
    user_template = config["user_template"]
    is_json = config["json_output"]

    results = []
    total = len(df)
    start_time = time.time()

    print(f"\nТехника: {technique}")
    print(f"Всего сообщений: {total}")
    print("-" * 50)

    for idx, row in df.iterrows():
        user_prompt = user_template.format(sms_text=row["text"])

        raw_response = query_llm(user_prompt, system_prompt)

        if is_json:
            reasoning, prediction = parse_json_response(raw_response)
        else:
            reasoning = raw_response
            prediction = parse_zero_shot_response(raw_response)

        results.append({
            "text": row["text"],
            "true_label": row["target"],
            "prediction": prediction,
            "reasoning": reasoning,
            "raw_response": raw_response,
        })

        done = len(results)
        if done % 50 == 0 or done == total:
            elapsed = time.time() - start_time
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / speed if speed > 0 else 0
            print(f"  [{done}/{total}] — {speed:.1f} msg/s, осталось ~{eta:.0f}с")

    result_df = pd.DataFrame(results)
    elapsed = time.time() - start_time
    print(f"Готово за {elapsed:.1f}с")

    return result_df


def main():
    """Точка входа — парсим аргументы и запускаем инференс."""
    parser = argparse.ArgumentParser(description="Инференс SMS через LLM")
    parser.add_argument(
        "--technique",
        choices=list(PROMPTS.keys()) + ["all"],
        default="all",
        help="Какую технику промптинга использовать (по умолчанию — все)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Сколько сообщений обработать (по умолчанию — все)",
    )
    parser.add_argument(
        "--data-path",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv"),
        help="Путь к датасету",
    )
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_path)
    print(f"Загружаем датасет: {data_path}")
    df = load_dataset(data_path)

    if args.limit:
        df = df.head(args.limit)
        print(f"Ограничиваем до {args.limit} сообщений")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    techniques = list(PROMPTS.keys()) if args.technique == "all" else [args.technique]

    for technique in techniques:
        result_df = run_inference(df, technique)

        output_path = os.path.join(RESULTS_DIR, f"{technique}_results.csv")
        result_df.to_csv(output_path, index=False)
        print(f"Результаты сохранены: {output_path}")

        valid = result_df[result_df["prediction"] >= 0]
        invalid = len(result_df) - len(valid)
        if len(valid) > 0:
            correct = (valid["prediction"] == valid["true_label"]).sum()
            print(f"  Accuracy (предварительная): {correct / len(valid):.3f}")
        if invalid > 0:
            print(f"  Не удалось распарсить: {invalid} ответов")

    print("\nИнференс завершён! Запусти evaluate.py для подробных метрик.")


if __name__ == "__main__":
    main()
