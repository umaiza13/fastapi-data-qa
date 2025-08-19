import os
import io
import json
import tempfile
import shutil
import base64
import matplotlib.pyplot as plt
from typing import Optional
from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://aipipe.org/openai/v1"
)

MAX_FILE_CHARS = 2000
OPENAI_TIMEOUT_SECONDS = 180


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


async def ask_gpt(question: str, files: dict = None, need_chart: bool = False) -> dict:
    if need_chart:
        system_prompt = (
            "You are a data analyst. "
            "The user asked for a chart/plot. "
            "Return JSON with: "
            "{answer: string, x: array, y: array, xlabel: string, ylabel: string, title: string}. "
            "Do not return code. Do not explain. Only valid JSON."
        )
    else:
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

    raw_output = response.choices[0].message.content.strip()

    import re
    raw_output = re.sub(r"^```(?:json)?|```$", "", raw_output, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw_output)
    except Exception:
        return {"error": "Invalid JSON from model", "raw_output": raw_output}


@app.post("/api/")
async def process_task(request: Request):
    form = await request.form()
    files = []
    main_question = None
    data_files = []
    temp_dir = tempfile.mkdtemp()

    try:
        # Collect all uploaded files
        for value in form.values():
            if isinstance(value, UploadFile):
                files.append(value)

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

        # 🔎 detect if chart requested
        keywords = ["chart", "plot", "graph", "visualization"]
        need_chart = any(k in main_question.lower() for k in keywords)

        llm_response = await ask_gpt(main_question, data_dict, need_chart=need_chart)

        if need_chart and "x" in llm_response and "y" in llm_response:
            # Create chart
            fig, ax = plt.subplots()
            ax.plot(llm_response["x"], llm_response["y"])
            ax.set_xlabel(llm_response.get("xlabel", ""))
            ax.set_ylabel(llm_response.get("ylabel", ""))
            ax.set_title(llm_response.get("title", ""))
            chart_b64 = fig_to_base64(fig)
            plt.close(fig)

            llm_response["chart"] = chart_b64

        return JSONResponse(content=llm_response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
