import os
import json
import zipfile
import shutil
import tempfile
import asyncio
import re
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
from openai import OpenAI

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://aipipe.org/openai/v1"
)

MAX_FILE_CHARS = 2000
OPENAI_TIMEOUT_SECONDS = 180  # 3 minutes

def summarize_file_content(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(MAX_FILE_CHARS)
            if len(content) == MAX_FILE_CHARS:
                content += "\n...[truncated]..."
            return content
    except Exception:
        size = os.path.getsize(file_path)
        return f"[Binary file: {os.path.basename(file_path)}, size: {size} bytes]"

def extract_url_from_text(text: str) -> str:
    match = re.search(r'https?://\S+', text)
    return match.group(0) if match else None

def scrape_table_from_url(url: str) -> str:
    response = requests.get(url)
    tables = pd.read_html(response.text)
    df = tables[0]
    df_head = df.head(20)
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "scraped_table.csv")
    df_head.to_csv(csv_path, index=False)
    return csv_path

async def ask_gpt(task: str, files: Optional[dict] = None) -> str:
    system_prompt = (
        "You are a data analyst who answers data questions. "
        "Always return the best possible answer as raw, syntactically valid JSON (object or array) with no explanation, no markdown, no code block, no extra text, and no trailing commas. "
        "If you do not have data files, answer using your general knowledge or best estimate. "
        "Do not include any comments or extra formatting. "
        "Your response MUST be valid JSON and nothing else."
    )

    prompt = task
    if files:
        summaries = []
        for name, content in files.items():
            if isinstance(content, bytes):
                try:
                    snippet = content.decode("utf-8")[:MAX_FILE_CHARS]
                    if len(snippet) == MAX_FILE_CHARS:
                        snippet += "\n...[truncated]..."
                    summaries.append(f"{name}:\n{snippet}")
                except Exception:
                    summaries.append(f"{name}: [binary or unreadable file]")
            else:
                snippet = content[:MAX_FILE_CHARS]
                if len(snippet) == MAX_FILE_CHARS:
                    snippet += "\n...[truncated]..."
                summaries.append(f"{name}:\n{snippet}")
        prompt += "\n\nAttached files:\n" + "\n".join(summaries)

    try:
        import concurrent.futures
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
        return response.choices[0].message.content.strip()
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="LLM request timed out after 3 minutes")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

import base64
import matplotlib.pyplot as plt
import io

def auto_stats_and_charts(csv_bytes, parsed_json, question):
    import pandas as pd
    from io import BytesIO
    df = pd.read_csv(BytesIO(csv_bytes))

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]

    for col in numeric_cols:
        parsed_json[f"average_{col}"] = round(df[col].mean(), 2)
        parsed_json[f"min_{col}"] = float(df[col].min())
        parsed_json[f"max_{col}"] = float(df[col].max())

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols[0]].corr(df[numeric_cols[1]])
        parsed_json[f"{numeric_cols[0]}_{numeric_cols[1]}_correlation"] = round(corr, 10)

    if "line chart" in question.lower() or "plot" in question.lower():
        x = df[date_cols[0]] if date_cols else df.index
        y = df[numeric_cols[0]]
        plt.figure(figsize=(4,3))
        plt.plot(x, y, color="blue")
        plt.xlabel(date_cols[0] if date_cols else "Index")
        plt.ylabel(numeric_cols[0])
        plt.title(f"{numeric_cols[0]} Line Chart")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        parsed_json[f"{numeric_cols[0]}_line_chart"] = base64.b64encode(buf.getvalue()).decode()

    if "histogram" in question.lower():
        plt.figure(figsize=(4,3))
        plt.hist(df[numeric_cols[0]], bins=10, color="orange")
        plt.xlabel(numeric_cols[0])
        plt.ylabel("Frequency")
        plt.title(f"{numeric_cols[0]} Histogram")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        parsed_json[f"{numeric_cols[0]}_histogram"] = base64.b64encode(buf.getvalue()).decode()

    if "scatterplot" in question.lower() and len(numeric_cols) >= 2:
        plt.figure(figsize=(4,3))
        plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], color="green")
        plt.xlabel(numeric_cols[0])
        plt.ylabel(numeric_cols[1])
        plt.title(f"{numeric_cols[0]} vs {numeric_cols[1]} Scatterplot")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        parsed_json[f"{numeric_cols[0]}_{numeric_cols[1]}_scatterplot"] = base64.b64encode(buf.getvalue()).decode()

    return parsed_json
import networkx as nx

def is_edge_list_csv(df):
    cols = [c.lower() for c in df.columns]
    return ("source" in cols and "target" in cols) or ("from" in cols and "to" in cols)

def network_stats_and_charts(csv_bytes, parsed_json, question):
    import pandas as pd
    from io import BytesIO
    import matplotlib.pyplot as plt
    import io, base64
    df = pd.read_csv(BytesIO(csv_bytes))
    G = nx.Graph()
    # Use first two columns as edges
    col1, col2 = df.columns[:2]
    for _, row in df.iterrows():
        G.add_edge(row[col1], row[col2])

    parsed_json['edge_count'] = G.number_of_edges()
    degrees = dict(G.degree())
    highest_degree_node = max(degrees, key=degrees.get)
    parsed_json['highest_degree_node'] = highest_degree_node
    parsed_json['average_degree'] = round(sum(degrees.values()) / len(degrees), 2)
    parsed_json['density'] = round(nx.density(G), 4)
    try:
        parsed_json['shortest_path_alice_eve'] = nx.shortest_path_length(G, 'Alice', 'Eve')
    except Exception:
        parsed_json['shortest_path_alice_eve'] = None

    # Draw network graph
    plt.figure(figsize=(4,4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    parsed_json['network_graph'] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # Degree histogram
    plt.figure(figsize=(4,3))
    degree_values = list(degrees.values())
    plt.bar(range(len(degree_values)), degree_values, color='green')
    plt.xlabel('Node')
    plt.ylabel('Degree')
    plt.title('Degree Distribution')
    plt.xticks(range(len(degrees)), list(degrees.keys()), rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    parsed_json['degree_histogram'] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return parsed_json

@app.post("/api/")
async def process_task(
    request: Request,
    question: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    temp_dir = tempfile.mkdtemp()
    main_question = None
    other_files = {}

    try:
        # Save uploaded files locally
        if files:
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                content_bytes = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content_bytes)

                # Accept both question.txt and questions.txt (case-insensitive)
                if file.filename.lower() in ["question.txt", "questions.txt"]:
                    try:
                        main_question = content_bytes.decode("utf-8")
                    except Exception:
                        raise HTTPException(status_code=400, detail="Could not decode question(s).txt as UTF-8")
                elif zipfile.is_zipfile(file_path):
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)
                        for name in zip_ref.namelist():
                            extracted_path = os.path.join(temp_dir, name)
                            try:
                                with open(extracted_path, "rb") as extracted_file:
                                    other_files[name] = extracted_file.read()
                            except Exception:
                                other_files[name] = b"[unreadable file]"
                else:
                    other_files[file.filename] = content_bytes

        # If no question(s).txt uploaded, fallback to typed question
        if not main_question:
            if question:
                main_question = question
            else:
                raise HTTPException(status_code=400, detail="No question provided: upload question.txt/questions.txt or use question form field")

        # Split questions by line, ignore empty lines
        if isinstance(main_question, str):
            questions_list = [q.strip() for q in main_question.splitlines() if q.strip()]
        else:
            questions_list = [main_question]

        # If only one question, return a single answer (not a list)
        is_single = len(questions_list) == 1

        results = []
        for q in questions_list:
            files_for_this_q = dict(other_files)
            # Web scraping feature for each question
            if "scrape" in q.lower():
                url = extract_url_from_text(q)
                if url:
                    try:
                        scraped_csv = scrape_table_from_url(url)
                        with open(scraped_csv, "rb") as f:
                            files_for_this_q["scraped_table.csv"] = f.read()
                    except Exception as e:
                        results.append({"error": f"Failed to scrape table from URL: {e}"})
                        continue

            llm_response = await ask_gpt(q, files_for_this_q if files_for_this_q else None)
            try:
                parsed_json = json.loads(llm_response)
            except json.JSONDecodeError:
                results.append({"error": f"LLM response not valid JSON: {llm_response}"})
                continue

            for fname, fbytes in files_for_this_q.items():
                if fname.endswith(".csv"):
                    parsed_json = auto_stats_and_charts(fbytes, parsed_json, q)
            results.append(parsed_json)

        # Return a single answer if only one question, else a list
        if is_single:
            return JSONResponse(content=results[0])
        else:
            return JSONResponse(content=results)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)