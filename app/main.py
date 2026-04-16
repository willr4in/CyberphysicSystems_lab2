"""
FastAPI-обёртка над Ollama сервером.

Один эндпоинт принимает запрос, прокидывает его в Ollama и возвращает ответ модели
"""

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="LLM Spam Detection Service",
    description="Прокси к Ollama для инференса Qwen2.5:0.5B",
    version="1.0.0",
)

OLLAMA_URL = "http://localhost:11434"


class GenerateRequest(BaseModel):
    """Что приходит от клиента."""

    prompt: str = Field(..., description="Текст запроса к модели")
    system: str = Field(default="", description="Системный промпт (если нужен)")
    temperature: float = Field(default=0.1, description="Температура генерации, ниже = стабильнее")
    model: str = Field(default="qwen2.5:0.5b", description="Название модели в Ollama")


class GenerateResponse(BaseModel):
    """Что уходит обратно клиенту."""

    response: str = Field(..., description="Текст ответа модели")
    model: str = Field(..., description="Какая модель отвечала")
    done: bool = Field(..., description="Завершена ли генерация")


@app.get("/health")
def health_check():
    """Проверка, что сервис жив."""
    return {"status": "ok"}


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Принимает запрос от клиента и пересылает его в Ollama.

    Работает просто: берём prompt и system из запроса,
    отправляем в Ollama, ждём полный ответ (stream=false)
    и возвращаем результат.
    """
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "system": request.system,
        "stream": False,
        "options": {
            "temperature": request.temperature,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            resp.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama сервер недоступен — возможно, ещё запускается",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Ollama не ответил вовремя — модель может быть перегружена",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Ollama вернул ошибку: {e.response.text}",
        )

    data = resp.json()

    return GenerateResponse(
        response=data.get("response", ""),
        model=data.get("model", request.model),
        done=data.get("done", True),
    )
