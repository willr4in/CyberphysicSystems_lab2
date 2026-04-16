"""
Считаем метрики по результатам инференса.

Для каждой техники промптинга считаем:
- Accuracy — доля правильных ответов
- Precision — из тех, кого модель назвала спамом, сколько реально спам
- Recall — из реального спама, сколько модель нашла
- F1 — гармоническое среднее precision и recall

Запуск:
    python research/evaluate.py
"""

import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Техники, которые оцениваем
TECHNIQUES = ["zero_shot", "cot", "few_shot", "cot_few_shot"]


def compute_metrics(true_labels: list[int], predictions: list[int]) -> dict:
    """
    Считаем все четыре метрики разом.

    Spam = positive class (1), ham = negative class (0).
    Если предсказаний нет — вернём нули, чтобы не упасть.
    """
    if not true_labels or not predictions:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, zero_division=0),
        "recall": recall_score(true_labels, predictions, zero_division=0),
        "f1": f1_score(true_labels, predictions, zero_division=0),
    }


def evaluate_technique(technique: str) -> dict | None:
    """
    Загружаем результаты конкретной техники и считаем метрики.

    Отбрасываем сообщения, где модель не смогла дать ответ (prediction = -1) (хотя можем и просто за ошибку).
    """
    results_path = os.path.join(RESULTS_DIR, f"{technique}_results.csv")

    if not os.path.exists(results_path):
        print(f"  Файл {results_path} не найден, пропускаем")
        return None

    df = pd.read_csv(results_path)
    total = len(df)

    valid = df[df["prediction"] >= 0].copy()
    invalid_count = total - len(valid)

    if len(valid) == 0:
        print(f"  Нет валидных предсказаний для {technique}")
        return None

    true_labels = valid["true_label"].tolist()
    predictions = valid["prediction"].tolist()

    metrics = compute_metrics(true_labels, predictions)
    metrics["total"] = total
    metrics["valid"] = len(valid)
    metrics["parse_errors"] = invalid_count
    metrics["parse_error_rate"] = invalid_count / total if total > 0 else 0

    return metrics


def pick_examples(technique: str, count: int = 5) -> list[dict] | None:
    """
    Выбираем несколько показательных примеров из результатов:
    - правильно найденный спам
    - правильно определённый ham
    - ошибка модели (если есть)

    Берём по чуть-чуть каждого типа, чтобы таблица была наглядной.
    """
    results_path = os.path.join(RESULTS_DIR, f"{technique}_results.csv")
    if not os.path.exists(results_path):
        return None

    df = pd.read_csv(results_path)
    valid = df[df["prediction"] >= 0].copy()
    if len(valid) == 0:
        return None

    examples = []
    label_map = {0: "ham", 1: "spam"}

    # Правильные спамы (true positive)
    tp = valid[(valid["true_label"] == 1) & (valid["prediction"] == 1)]
    for _, row in tp.head(2).iterrows():
        examples.append(row)

    # Правильные ham (true negative)
    tn = valid[(valid["true_label"] == 0) & (valid["prediction"] == 0)]
    for _, row in tn.head(2).iterrows():
        examples.append(row)

    # Ошибки (любые)
    errors = valid[valid["true_label"] != valid["prediction"]]
    for _, row in errors.head(1).iterrows():
        examples.append(row)

    result = []
    for row in examples[:count]:
        text = str(row["text"])[:60] + "..." if len(str(row["text"])) > 60 else str(row["text"])
        # Экранируем пайпы, чтобы не ломали markdown-таблицу
        text = text.replace("|", "\\|")
        reasoning = str(row.get("reasoning", ""))
        reasoning = reasoning[:80] + "..." if len(reasoning) > 80 else reasoning
        reasoning = reasoning.replace("|", "\\|")

        result.append({
            "text": text,
            "true": label_map.get(int(row["true_label"]), "?"),
            "pred": label_map.get(int(row["prediction"]), "?"),
            "match": "+" if row["true_label"] == row["prediction"] else "x",
            "reasoning": reasoning,
        })

    return result


def format_report(all_metrics: dict[str, dict]) -> str:
    """
    Формируем красивый текстовый отчёт с таблицей метрик.

    Этот отчёт пойдёт в report.md и будет выведен в консоль.
    """
    lines = []
    lines.append("# Отчёт: сравнение техник промптинга для SMS Spam Detection")
    lines.append("")
    lines.append("## Модель: Qwen2.5:0.5B (Ollama)")
    lines.append("")

    sample_size = 0
    for m in all_metrics.values():
        if m:
            sample_size = m["total"]
            break
    lines.append(f"**Размер выборки:** {sample_size} SMS из датасета SMS Spam Collection")
    lines.append("")
    lines.append("## Результаты")
    lines.append("")

    header = "| Техника | Accuracy | Precision | Recall | F1 | Обработано | Ошибки парсинга |"
    separator = "|---------|----------|-----------|--------|------|------------|-----------------|"
    lines.append(header)
    lines.append(separator)

    for technique in TECHNIQUES:
        m = all_metrics.get(technique)
        if m is None:
            lines.append(f"| {technique} | — | — | — | — | — | — |")
            continue

        lines.append(
            f"| {technique} "
            f"| {m['accuracy']:.4f} "
            f"| {m['precision']:.4f} "
            f"| {m['recall']:.4f} "
            f"| {m['f1']:.4f} "
            f"| {m['valid']}/{m['total']} "
            f"| {m['parse_errors']} ({m['parse_error_rate']:.1%}) |"
        )

    lines.append("")

    best_technique = None
    best_f1 = -1
    for technique, m in all_metrics.items():
        if m and m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_technique = technique

    if best_technique:
        examples = pick_examples(best_technique)
        if examples:
            lines.append(f"## Примеры предсказаний ({best_technique})")
            lines.append("")
            lines.append("| SMS (начало) | Реальный | Модель | | Reasoning |")
            lines.append("|---|---|---|---|---|")
            for ex in examples:
                lines.append(
                    f"| {ex['text']} "
                    f"| {ex['true']} "
                    f"| {ex['pred']} "
                    f"| {ex['match']} "
                    f"| {ex['reasoning']} |"
                )
            lines.append("")

    lines.append("## Примеры по каждой технике")
    lines.append("")
    for technique in TECHNIQUES:
        if technique not in all_metrics:
            continue
        examples = pick_examples(technique, count=3)
        if not examples:
            continue
        lines.append(f"### {technique}")
        lines.append("")
        lines.append("| SMS (начало) | Реальный | Модель | | Reasoning |")
        lines.append("|---|---|---|---|---|")
        for ex in examples:
            lines.append(
                f"| {ex['text']} "
                f"| {ex['true']} "
                f"| {ex['pred']} "
                f"| {ex['match']} "
                f"| {ex['reasoning']} |"
            )
        lines.append("")

    lines.append("## Анализ")
    lines.append("")

    best_technique = None
    best_f1 = -1
    for technique, m in all_metrics.items():
        if m and m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_technique = technique

    if best_technique:
        lines.append(f"**Лучшая техника по F1:** {best_technique} (F1 = {best_f1:.4f})")
        lines.append("")

    lines.append("### Выводы")
    lines.append("")
    lines.append("1. **Zero-shot** — базовый бенчмарк без подсказок. Модель опирается ")
    lines.append("   только на своё понимание задачи из предобучения.")
    lines.append("")
    lines.append("2. **Chain-of-Thought** — пошаговое рассуждение помогает модели ")
    lines.append("   не торопиться с выводом и учитывать больше признаков.")
    lines.append("")
    lines.append("3. **Few-shot** — примеры в промпте задают модели формат и паттерн, ")
    lines.append("   показывая, на что обращать внимание.")
    lines.append("")
    lines.append("4. **CoT + Few-shot** — комбинация обеих техник: модель видит примеры ")
    lines.append("   с рассуждениями и повторяет тот же аналитический процесс.")
    lines.append("")
    lines.append("### Замечания")
    lines.append("")
    lines.append("- Датасет несбалансирован (~87% ham, ~13% spam), поэтому accuracy ")
    lines.append("  может быть обманчиво высокой. F1 — более надёжная метрика.")
    lines.append("- Qwen2.5:0.5B — компактная модель. Более крупные модели (3B, 7B) ")
    lines.append("  показали бы существенно лучшие результаты.")
    lines.append("- Ошибки парсинга (когда модель не смогла вернуть нужный формат) ")
    lines.append("  тоже важны — они показывают, насколько хорошо модель следует инструкциям.")
    lines.append("")

    return "\n".join(lines)


def main():
    """Основной цикл — оцениваем каждую технику и выводим отчёт."""
    print("Оценка техник промптинга\n")

    if not os.path.exists(RESULTS_DIR):
        print(f"Папка {RESULTS_DIR} не найдена. Сначала запусти inference.py!")
        sys.exit(1)

    all_metrics = {}
    for technique in TECHNIQUES:
        print(f"\nТехника: {technique}")
        metrics = evaluate_technique(technique)
        if metrics:
            all_metrics[technique] = metrics
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  Ошибки парсинга: {metrics['parse_errors']}/{metrics['total']}")

    if not all_metrics:
        print("\nНет результатов для оценки!")
        sys.exit(1)

    # Генерируем отчёт
    report = format_report(all_metrics)

    report_path = os.path.join(os.path.dirname(__file__), "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nОтчёт сохранён: {report_path}")

    # Сводную таблицу тоже сохраним в CSV — пригодится
    summary_df = pd.DataFrame(all_metrics).T
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    summary_df.to_csv(summary_path)
    print(f"Сводная таблица: {summary_path}")


if __name__ == "__main__":
    main()
