import io
import os
import shutil
import tempfile
import base64
import pandas as pd
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse

from llm import ask_gpt  # <-- your LLM call via Sanand’s AI Pipe

app = FastAPI()


def fig_to_base64(fig):
    """Convert Matplotlib figure to base64 PNG under 100kB."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    data = buf.getvalue()

    # Resize until under 100kB
    if len(data) > 100_000:
        fig.set_size_inches(4, 3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        data = buf.getvalue()

    return base64.b64encode(data).decode("utf-8")


@app.post("/api/")
async def process_task(request: Request):
    temp_dir = tempfile.mkdtemp()
    main_question = None
    data_files = []

    try:
        form = await request.form()
        if not form:
            raise HTTPException(status_code=400, detail="No files provided.")

        # Accept all fields (no matter the name)
        for field_name, field_value in form.items():
            if isinstance(field_value, UploadFile):
                fname_lower = field_value.filename.lower()
                content_bytes = await field_value.read()

                if fname_lower in ("question.txt", "questions.txt") and main_question is None:
                    try:
                        main_question = content_bytes.decode("utf-8").strip()
                    except Exception:
                        raise HTTPException(status_code=400, detail="Question file not readable.")
                else:
                    data_files.append((field_value.filename, content_bytes))

        if not main_question:
            raise HTTPException(status_code=400, detail="No question.txt or questions.txt file provided.")

        # Prepare context for LLM
        data_dict = {fname: content for fname, content in data_files} if data_files else None

        # Call LLM
        llm_response = await ask_gpt(main_question, data_dict)

        # Auto-generate charts if asked
        if any(word in main_question.lower() for word in ["chart", "plot", "graph"]):
            for fname, content in data_files:
                if fname.lower().endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(content))

                    if "temp" in df.columns and "date" in df.columns:
                        fig, ax = plt.subplots()
                        df.plot(x="date", y="temp", ax=ax, color="red")
                        llm_response["temp_line_chart"] = fig_to_base64(fig)

                    if "precip" in df.columns:
                        fig, ax = plt.subplots()
                        df["precip"].plot(kind="hist", bins=20, ax=ax, color="orange")
                        llm_response["precip_histogram"] = fig_to_base64(fig)

        return JSONResponse(content=llm_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
