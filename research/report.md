# Отчёт: сравнение техник промптинга для SMS Spam Detection

## Модель: Qwen2.5:0.5B (Ollama)

**Размер выборки:** 100 SMS из датасета SMS Spam Collection

## Результаты

| Техника | Accuracy | Precision | Recall | F1 | Обработано | Ошибки парсинга |
|---------|----------|-----------|--------|------|------------|-----------------|
| zero_shot | 0.7600 | 0.1111 | 0.0588 | 0.0769 | 100/100 | 0 (0.0%) |
| cot | 0.9000 | 0.8889 | 0.4706 | 0.6154 | 100/100 | 0 (0.0%) |
| few_shot | 0.7600 | 0.4054 | 0.8824 | 0.5556 | 100/100 | 0 (0.0%) |
| cot_few_shot | 0.8500 | 1.0000 | 0.1176 | 0.2105 | 100/100 | 0 (0.0%) |

## Примеры предсказаний (cot)

| SMS (начало) | Реальный | Модель | | Reasoning |
|---|---|---|---|---|
| Had your mobile 11 months or more? U R entitled to Update to... | spam | spam | + | The language is informal and uses exclamation marks, which are common in spam me... |
| SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH1... | spam | spam | + | The message contains a clear offer for cash prizes and is in all caps with an ex... |
| Go until jurong point, crazy.. Available only in bugis n gre... | ham | ham | + | The language used is not urgent or all caps, and the exclamation marks are not p... |
| Ok lar... Joking wif u oni... | ham | ham | + | The language used in the message does not contain any urgency, ALL CAPS, exclama... |
| Free entry in 2 a wkly comp to win FA Cup final tkts 21st Ma... | spam | ham | x | The message contains a link to a website, which is not typical of spam messages. |

## Примеры по каждой технике

### zero_shot

| SMS (начало) | Реальный | Модель | | Reasoning |
|---|---|---|---|---|
| XXXMobileMovieClub: To use your credit, click the WAP link i... | spam | spam | + | spam |
| Ok lar... Joking wif u oni... | ham | ham | + | ham |
| U dun say so early hor... U c already then say... | ham | ham | + | ham |

### cot

| SMS (начало) | Реальный | Модель | | Reasoning |
|---|---|---|---|---|
| Had your mobile 11 months or more? U R entitled to Update to... | spam | spam | + | The language is informal and uses exclamation marks, which are common in spam me... |
| SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH1... | spam | spam | + | The message contains a clear offer for cash prizes and is in all caps with an ex... |
| Go until jurong point, crazy.. Available only in bugis n gre... | ham | ham | + | The language used is not urgent or all caps, and the exclamation marks are not p... |

### few_shot

| SMS (начало) | Реальный | Модель | | Reasoning |
|---|---|---|---|---|
| Free entry in 2 a wkly comp to win FA Cup final tkts 21st Ma... | spam | spam | + | Free competition entry, asks to text a short code, promotional language |
| FreeMsg Hey there darling it's been 3 week's now and no word... | spam | spam | + | Spam |
| Go until jurong point, crazy.. Available only in bugis n gre... | ham | ham | + | Casual message between friends about meeting up |

### cot_few_shot

| SMS (начало) | Реальный | Модель | | Reasoning |
|---|---|---|---|---|
| Urgent UR awarded a complimentary trip to EuroDisinc Trav, A... | spam | spam | + | Step 1: Urgent and promotional tone with 'UR awarded a complimentary trip'. Step... |
| Please call our customer service representative on 0800 169 ... | spam | spam | + | Step 1: Urgent and promotional tone with 'WON' language. Step 2: No spam signals... |
| Go until jurong point, crazy.. Available only in bugis n gre... | ham | ham | + | Step 1: Casual tone with slang ('la', 'crazy..'). Step 2: No spam signals. Step ... |

## Анализ

**Лучшая техника по F1:** cot (F1 = 0.6154)

### Выводы

1. **Zero-shot** — базовый бенчмарк без подсказок. Модель опирается 
   только на своё понимание задачи из предобучения.

2. **Chain-of-Thought** — пошаговое рассуждение помогает модели 
   не торопиться с выводом и учитывать больше признаков.

3. **Few-shot** — примеры в промпте задают модели формат и паттерн, 
   показывая, на что обращать внимание.

4. **CoT + Few-shot** — комбинация обеих техник: модель видит примеры 
   с рассуждениями и повторяет тот же аналитический процесс.

### Замечания

- Датасет несбалансирован (~87% ham, ~13% spam), поэтому accuracy 
  может быть обманчиво высокой. F1 — более надёжная метрика.
- Qwen2.5:0.5B — компактная модель. Более крупные модели (3B, 7B) 
  показали бы существенно лучшие результаты.
- Ошибки парсинга (когда модель не смогла вернуть нужный формат) 
  тоже важны — они показывают, насколько хорошо модель следует инструкциям.
