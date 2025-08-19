# llm.py
import os
import httpx

# Sanand’s AI Pipe proxy URL
AI_PIPE_URL = "https://aipipe.openai-proxy.com/v1/chat/completions"

# Get your token from environment variables (set in Render dashboard)
AI_PIPE_KEY = os.getenv("OPENAI_API_KEY")

async def ask_gpt(prompt: str, context: str = "") -> str:
    """
    Send a prompt + optional context to the AI Pipe proxy and return the LLM's answer.
    """
    if not AI_PIPE_KEY:
        raise RuntimeError("AI_PIPE_KEY is not set in environment variables")

    headers = {
        "Authorization": f"Bearer {AI_PIPE_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-4o-mini",  # or whichever model Sanand specified
        "messages": [
            {"role": "system", "content": "You are a helpful data analyst assistant."},
            {"role": "user", "content": prompt + ("\n\n" + context if context else "")},
        ],
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(AI_PIPE_URL, headers=headers, json=data)
        resp.raise_for_status()
        result = resp.json()

    return result["choices"][0]["message"]["content"]
