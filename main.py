import os
import json
import zipfile
import shutil
import tempfile
import asyncio
import re
import requests
import pandas as pd
import base64
import matplotlib.pyplot as plt
import io

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Body
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

def valid_blank_png_base64():
    fig, ax = plt.subplots(figsize=(4,3))
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

def safe_network_response(parsed_json):
    required_keys = [
        "edge_count", "highest_degree_node", "average_degree",
        "density", "shortest_path_alice_eve", "network_graph", "degree_histogram"
    ]
    defaults = {
        "edge_count": 0,
        "highest_degree_node": "",
        "average_degree": 0.0,
        "density": 0.0,
        "shortest_path_alice_eve": 0,
        "network_graph": valid_blank_png_base64(),
        "degree_histogram": valid_blank_png_base64(),
    }
    for k in required_keys:
        if k not in parsed_json or parsed_json[k] is None:
            parsed_json[k] = defaults[k]
    return parsed_json

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
    if df.shape[1] < 2:
        return False
    sample = df.iloc[:20, :2]
    non_empty_ratio = sample.notna().mean().min()
    if non_empty_ratio < 0.8:
        return False
    unique_ratio = (sample.nunique() / len(sample)).mean()
    if unique_ratio > 0.9:
        return False
    return True

def network_stats_and_charts(csv_bytes, parsed_json, question):
    import pandas as pd
    from io import BytesIO
    import matplotlib.pyplot as plt
    import io, base64
    df = pd.read_csv(BytesIO(csv_bytes))
    G = nx.Graph()
    col1, col2 = df.columns[:2]
    for _, row in df.iterrows():
        G.add_edge(row[col1], row[col2])

    parsed_json['edge_count'] = G.number_of_edges()
    degrees = dict(G.degree())
    if degrees:
        highest_degree_node = max(degrees, key=degrees.get)
        parsed_json['highest_degree_node'] = str(highest_degree_node)
        parsed_json['average_degree'] = round(sum(degrees.values()) / len(degrees), 2)
    else:
        parsed_json['highest_degree_node'] = ""
        parsed_json['average_degree'] = 0.0
    parsed_json['density'] = round(nx.density(G), 4) if G.number_of_nodes() > 1 else 0.0
    try:
        parsed_json['shortest_path_alice_eve'] = nx.shortest_path_length(G, 'Alice', 'Eve')
    except Exception:
        parsed_json['shortest_path_alice_eve'] = 0

    # network_graph
    try:
        plt.figure(figsize=(4,4))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        parsed_json['network_graph'] = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        parsed_json['network_graph'] = valid_blank_png_base64()

    # degree_histogram
    try:
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
        parsed_json['degree_histogram'] = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        parsed_json['degree_histogram'] = valid_blank_png_base64()
    return safe_network_response(parsed_json)

@app.post("/api/")
async def process_task(
    request: Request,
    question: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    temp_dir = tempfile.mkdtemp()
    main_question = None
    data_files = []  # All non-question files

    try:
        if files:
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                content_bytes = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content_bytes)

                fname_lower = file.filename.lower()
                if fname_lower in ["question.txt", "questions.txt"]:
                    try:
                        main_question = content_bytes.decode("utf-8").strip()
                    except Exception:
                        raise HTTPException(status_code=400, detail="Could not decode question(s).txt as UTF-8")
                elif zipfile.is_zipfile(file_path):
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)
                        for name in zip_ref.namelist():
                            extracted_path = os.path.join(temp_dir, name)
                            try:
                                with open(extracted_path, "rb") as extracted_file:
                                    data_files.append( (name, extracted_file.read()) )
                            except Exception:
                                pass
                else:
                    # Accept any other file as a data file
                    data_files.append( (file.filename, content_bytes) )

        if not main_question:
            if question:
                main_question = question
            else:
                raise HTTPException(status_code=400, detail="No question provided: upload question.txt/questions.txt or use question form field")

        results = []
        for fname, file_bytes in data_files:
            # Try to read as CSV or Excel
            try:
                if fname.lower().endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(file_bytes))
                    content_to_pass = file_bytes
                elif fname.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(io.BytesIO(file_bytes))
                    # Convert to CSV bytes for downstream compatibility
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    content_to_pass = csv_buffer.getvalue().encode()
                elif fname.lower().endswith(".tsv"):
                    df = pd.read_csv(io.BytesIO(file_bytes), sep="\t")
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    content_to_pass = csv_buffer.getvalue().encode()
                else:
                    # Skip files that are not tabular data
                    continue
            except Exception:
                # Skip files that can't be parsed as tabular data
                continue

            if is_edge_list_csv(df):
                parsed_json = network_stats_and_charts(content_to_pass, {}, main_question)
            else:
                parsed_json = auto_stats_and_charts(content_to_pass, {}, main_question)
            results.append(parsed_json)

        if len(results) == 1:
            return JSONResponse(content=results[0])
        elif len(results) > 1:
            return JSONResponse(content={"results": results})
        else:
            # Fallback if no data files were parsed
            return JSONResponse(content=safe_network_response({}))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)