"""
Шаблоны промптов для классификации SMS спама.

Четыре техники:
1. zero_shot — просто спроси модель, спам или нет
2. cot — попросим рассуждать по шагам (Chain-of-Thought)
3. few_shot — покажем примеры, чтобы модель поняла паттерн
4. cot_few_shot — и примеры, и рассуждение вместе

Для CoT, few-shot и CoT+few-shot модель должна отвечать
строго в JSON: {"reasoning": "...", "verdict": 0 или 1}
"""


# Zero-shot 
# Тут всё просто: никаких примеров, никаких подсказок.
# Просто говорим модели, что делать.

ZERO_SHOT_SYSTEM = (
    "You are an SMS spam classifier. "
    "For each SMS message, reply with exactly one word: spam or ham. "
    "spam means the message is unsolicited advertising or scam. "
    "ham means it is a normal, legitimate message."
)

ZERO_SHOT_USER = "Classify this SMS:\n{sms_text}"


# Chain-of-Thought (CoT) 
# Просим модель подумать перед ответом — часто помогает
# маленьким моделям не торопиться с выводом.

COT_SYSTEM = """You are an SMS spam classifier. Your task is to determine whether an SMS message is spam (1) or ham (0).

Think step by step before giving your verdict:
1. Look at the language: does it use urgency, ALL CAPS, exclamation marks?
2. Check for typical spam indicators: prizes, free offers, suspicious links, phone numbers to call.
3. Consider if this looks like a normal conversation between people.

You MUST respond with valid JSON only, no other text:
{"reasoning": "<your step-by-step analysis>", "verdict": <0 or 1>}

Where verdict is: 0 = ham (not spam), 1 = spam."""

COT_USER = "Analyze this SMS message:\n{sms_text}"


# Few-shot
# Даём модели несколько примеров — она видит паттерн и повторяет.
# Примеры подбирал так, чтобы показать типичный спам и типичный ham.

FEW_SHOT_SYSTEM = """You are an SMS spam classifier. Classify each SMS as spam (1) or ham (0).

Here are examples:

SMS: "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."
{"reasoning": "Casual message between friends about meeting up, no commercial intent", "verdict": 0}

SMS: "WINNER!! As a valued network customer you have been selected to receivea 900 prize reward! To claim call 09061701461."
{"reasoning": "Prize notification with phone number to call, classic spam pattern", "verdict": 1}

SMS: "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k?"
{"reasoning": "Personal conversation, emotional tone, clearly between two people who know each other", "verdict": 0}

SMS: "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)"
{"reasoning": "Free competition entry, asks to text a short code, promotional language", "verdict": 1}

SMS: "Nah I don't think he goes to usf, he lives around here though"
{"reasoning": "Casual chat about someone they both know, completely normal conversation", "verdict": 0}

Now classify the following SMS. You MUST respond with valid JSON only, no other text:
{"reasoning": "<brief explanation>", "verdict": <0 or 1>}"""

FEW_SHOT_USER = "SMS: \"{sms_text}\""


# CoT + Few-shot 
# Комбинируем обе техники: и примеры с рассуждениями, и инструкция думать по шагам.
# По идее, это должно дать лучший результат.

COT_FEW_SHOT_SYSTEM = """You are an SMS spam classifier. Determine if an SMS is spam (1) or ham (0).

Follow this analysis process for each message:
Step 1: Identify the tone (casual/formal/promotional/urgent)
Step 2: Look for spam signals (prizes, free offers, urgent calls-to-action, suspicious numbers/links)
Step 3: Look for ham signals (personal names, conversational flow, specific shared context)
Step 4: Make your decision based on the evidence

Here are examples with step-by-step reasoning:

SMS: "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."
{"reasoning": "Step 1: Casual tone with slang ('la', 'crazy..'). Step 2: No spam signals. Step 3: Discussing specific places to meet, conversational style. Step 4: Normal chat between friends.", "verdict": 0}

SMS: "WINNER!! As a valued network customer you have been selected to receivea 900 prize reward! To claim call 09061701461."
{"reasoning": "Step 1: Urgent and promotional tone with ALL CAPS 'WINNER'. Step 2: Prize reward, phone number to call, 'selected' language — classic spam. Step 3: No personal context. Step 4: Clearly spam.", "verdict": 1}

SMS: "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k?"
{"reasoning": "Step 1: Casual and emotional tone. Step 2: No spam indicators at all. Step 3: Personal conversation with shared context ('this stuff', 'tonight'). Step 4: Ham.", "verdict": 0}

SMS: "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)"
{"reasoning": "Step 1: Promotional tone. Step 2: 'FREE entry', competition, short code 87121, asks to text. Step 3: No personal elements. Step 4: Spam.", "verdict": 1}

SMS: "Nah I don't think he goes to usf, he lives around here though"
{"reasoning": "Step 1: Casual tone. Step 2: No spam signals. Step 3: Discussing a mutual acquaintance, conversational. Step 4: Ham.", "verdict": 0}

Now classify the following SMS using the same step-by-step process. You MUST respond with valid JSON only, no other text:
{"reasoning": "<step-by-step analysis>", "verdict": <0 or 1>}"""

COT_FEW_SHOT_USER = "SMS: \"{sms_text}\""


PROMPTS = {
    "zero_shot": {
        "system": ZERO_SHOT_SYSTEM,
        "user_template": ZERO_SHOT_USER,
        "json_output": False,  # zero-shot отвечает текстом, не JSON
    },
    "cot": {
        "system": COT_SYSTEM,
        "user_template": COT_USER,
        "json_output": True,
    },
    "few_shot": {
        "system": FEW_SHOT_SYSTEM,
        "user_template": FEW_SHOT_USER,
        "json_output": True,
    },
    "cot_few_shot": {
        "system": COT_FEW_SHOT_SYSTEM,
        "user_template": COT_FEW_SHOT_USER,
        "json_output": True,
    },
}
