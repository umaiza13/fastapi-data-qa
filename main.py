import os
import io
import json
import tempfile
import shutil
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://aipipe.org/openai/v1"
)

MAX_FILE_CHARS = 2000
OPENAI_TIMEOUT_SECONDS = 180

async def ask_gpt(question: str, files: dict = None) -> dict:
    system_prompt = (
        "You are a data analyst. "
        "Given the question and any attached data files, return the best possible answer as valid JSON. "
        "Do not explain. Only return JSON (object or array)."
    )
    prompt = question
    if files:
        summaries = []
        for name, content in files.items():
            try:
                snippet = content.decode("utf-8")[:MAX_FILE_CHARS]
            except Exception:
                snippet = "[binary file]"
            summaries.append(f"{name}:\n{snippet}")
        prompt += "\n\nAttached files:\n" + "\n".join(summaries)
    import asyncio, concurrent.futures
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        response = await asyncio.wait_for(
            loop.run_in_executor(
                pool,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
            ),
            timeout=OPENAI_TIMEOUT_SECONDS
        )
    return json.loads(response.choices[0].message.content.strip())

@app.post("/api/")
async def process_task(
    request: Request,
    files: Optional[List[UploadFile]] = File(None)
):
    temp_dir = tempfile.mkdtemp()
    main_question = None
    data_files = []

    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided.")

        # Identify main question file and data files
        for file in files:
            fname_lower = file.filename.lower()
            content_bytes = await file.read()
            if fname_lower in ("question.txt", "questions.txt") and main_question is None:
                try:
                    main_question = content_bytes.decode("utf-8").strip()
                except Exception:
                    raise HTTPException(status_code=400, detail="Question file not readable.")
            else:
                data_files.append((file.filename, content_bytes))

        if not main_question:
            raise HTTPException(status_code=400, detail="No question.txt or questions.txt file provided.")

        data_dict = {fname: content for fname, content in data_files} if data_files else None

        llm_response = await ask_gpt(main_question, data_dict)
        return JSONResponse(content=llm_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)