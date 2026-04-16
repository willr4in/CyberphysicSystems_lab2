# Лабораторная работа №2: SMS Spam Detection via LLM — NLP proof-of-concept

## ## Задание на четвёрку
**Курс:** Киберфизические системы  

Proof-of-concept прототип распознавания спама в SMS с помощью LLM (Qwen2.5:0.5B).

Заказчик — провайдер сотовой связи. В рамках повышения удержания пользователей открыта
инициатива исследования борьбы со спамом в СМС посредством LLM технологий.

Данный прототип демонстрирует работу модели Qwen2.5:0.5B на задаче бинарной классификации
SMS-сообщений (spam / ham) с использованием различных техник промптинга.

## Архитектура

```
┌──────────────────────────────────────────────┐
│              Docker Container                │
│                                              │
│   Ollama (Qwen2.5:0.5B)  ◄──►  FastAPI      │
│       :11434                    :8000        │
└──────────────────────┬───────────────────────┘
                       │ port 8000
                  ┌────┴────┐
                  │   Host  │
                  │ scripts │
                  └─────────┘
```

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone <repo-url>
cd CyberphysicSystems_lab2
```

### 2. Скачать датасет

Скачайте [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
и поместите файл `spam.csv` в папку `data/`.

### 3. Создать виртуальное окружение (для хост-скриптов)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Запустить LLM-сервис

```bash
docker compose up --build -d
```

### 5. Проверить работоспособность

```bash
# Через curl
bash scripts/test_service.sh

# Через Python
python scripts/test_service.py
```

### 6. Запустить исследование (дополнительная часть)

```bash
# Инференс по всем техникам промптинга (весь датасет)
python research/inference.py

# Или сначала на малой выборке для проверки
python research/inference.py --limit 100

# Можно запустить только одну технику
python research/inference.py --technique cot --limit 200

# Подсчёт метрик и генерация отчёта
python research/evaluate.py
```

**Как это работает:** `inference.py` прогоняет SMS через модель и сохраняет
результаты в `research/results/`. Затем `evaluate.py` читает эти результаты,
считает метрики и перезаписывает `research/report.md`. При каждом новом запуске
инференса результаты и отчёт обновляются под текущую выборку.

**Полные результаты:** после инференса в `research/results/` появятся CSV-файлы
для каждой техники (`zero_shot_results.csv`, `cot_results.csv` и т.д.).
В них — текст каждого SMS, реальная метка, предсказание модели, reasoning
и сырой ответ. Можно открыть и посмотреть, как модель отработала на каждом сообщении.

Итоговый отчёт с таблицами метрик, примерами и выводами — [`research/report.md`](research/report.md).

## Стек

| Компонент | Назначение |
|-----------|-----------|
| Docker | Контейнеризация LLM-сервиса |
| Ollama | Сервер для запуска LLM |
| Qwen2.5:0.5B | Языковая модель для классификации |
| FastAPI | HTTP-обёртка над Ollama |
| scikit-learn | Метрики оценки |
| pandas | Работа с датасетом |

## Структура проекта

```
├── docker/
│   ├── Dockerfile             # Образ: Ubuntu 22.04 + Python + Ollama
│   └── entrypoint.sh          # Старт ollama -> загрузка модели -> FastAPI
├── docker-compose.yml         # Оркестрация, проброс портов
├── app/
│   └── main.py                # FastAPI сервис
├── scripts/
│   ├── test_ollama.sh         # Тест ollama через curl (внутри контейнера)
│   ├── test_service.sh        # Тест FastAPI через curl (с хоста)
│   └── test_service.py        # Тест FastAPI через requests (с хоста)
├── research/
│   ├── prompts.py             # Шаблоны промптов
│   ├── inference.py           # Скрипт инференса на датасете
│   ├── evaluate.py            # Подсчёт метрик
│   └── report.md              # Отчёт исследования
├── data/
│   └── spam.csv               # Датасет (скачать с Kaggle)
├── requirements.txt           # Зависимости хоста
└── requirements-docker.txt    # Зависимости контейнера
```

## Техники промптинга

| Техника | Описание |
|---------|----------|
| Zero-shot | Прямая инструкция без примеров |
| Chain-of-Thought (CoT) | Пошаговое рассуждение перед ответом |
| Few-shot | Несколько примеров spam/ham в промпте |
| CoT + Few-shot | Примеры с пошаговым рассуждением |
