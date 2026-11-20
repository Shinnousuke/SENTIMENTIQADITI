from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import pandas as pd, io, sqlite3, json, os
from pathlib import Path
from .ml_engine import build_analysis, semantic_search, answer_question


import csv
import json
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse



ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "feedback.sqlite3"
app = FastAPI(title="SentimentIQ Pro API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATE = {"analysis": None, "filename": None}


def init_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as con:
        con.execute("CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, review TEXT NOT NULL, predicted TEXT, corrected TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
init_db()

class SearchRequest(BaseModel): query: str
class AskRequest(BaseModel): question: str
class FeedbackRequest(BaseModel): review: str; predicted: str | None = None; corrected: str

@app.get("/api/health")
def health(): return {"status":"ok", "service":"SentimentIQ Pro", "analysis_loaded": STATE["analysis"] is not None}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file.")
    raw = await file.read()
    if len(raw) > 25 * 1024 * 1024: raise HTTPException(413, "CSV must be smaller than 25 MB.")
    try:
        try: df = pd.read_csv(io.BytesIO(raw))
        except UnicodeDecodeError: df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
        result = build_analysis(df)
        STATE["analysis"] = result
        STATE["filename"] = file.filename
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(422, str(e))

@app.post("/api/search")
def search(body: SearchRequest):
    if not STATE["analysis"]: raise HTTPException(400, "Upload a CSV first.")
    return {"results": semantic_search(STATE["analysis"]["reviews"], body.query)}

@app.post("/api/ask")
def ask(body: AskRequest):
    if not STATE["analysis"]: raise HTTPException(400, "Upload a CSV first.")
    return answer_question(STATE["analysis"]["reviews"], body.question)

@app.post("/api/feedback")
def feedback(body: FeedbackRequest):
    if body.corrected not in ["Positive", "Neutral", "Negative"]: raise HTTPException(400, "Invalid corrected sentiment.")
    with sqlite3.connect(DB) as con:
        con.execute("INSERT INTO feedback(review,predicted,corrected) VALUES(?,?,?)", (body.review, body.predicted, body.corrected))
    return {"saved": True, "message": "Correction saved to the local feedback database. Export it for a future retraining run."}

@app.get("/api/feedback/export")
def export_feedback():
    with sqlite3.connect(DB) as con: df = pd.read_sql_query("SELECT review,predicted,corrected,created_at FROM feedback", con)
    return Response(df.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition":"attachment; filename=sentimentiq_feedback.csv"})

@app.get("/api/export")
def export_results():
    if not STATE["analysis"]: raise HTTPException(400, "Upload a CSV first.")
    df = pd.DataFrame([{k:r[k] for k in ["index","review","sentiment","confidence","uncertain"]} for r in STATE["analysis"]["reviews"]])
    return Response(df.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition":"attachment; filename=sentimentiq_predictions.csv"})




# ==========================================================
# MULTILINGUAL INDIC SENTIMENT EVALUATION API
# ==========================================================

INDIC_DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "indic_evaluation"
)


def read_csv_file(filename):
    file_path = INDIC_DATA_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation file not found: {filename}"
        )

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


@app.get("/api/indic-evaluation/summary")
def indic_evaluation_summary():

    file_path = INDIC_DATA_DIR / "metrics.json"

    if not file_path.exists():
        raise HTTPException(404, "metrics.json not found")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/api/indic-evaluation/languages")
def indic_language_metrics():

    return read_csv_file("language_wise_metrics.csv")


@app.get("/api/indic-evaluation/predictions")
def indic_all_predictions(limit: int = 100, offset: int = 0):

    rows = read_csv_file("all_predictions.csv")

    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    return {
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "results": rows[offset:offset + limit]
    }


@app.get("/api/indic-evaluation/errors")
def indic_incorrect_predictions():

    rows = read_csv_file("incorrect_predictions.csv")

    return {
        "total": len(rows),
        "results": rows[:500]
    }


@app.get("/api/indic-evaluation/charts")
def indic_chart_list():

    chart_dir = INDIC_DATA_DIR / "charts"

    if not chart_dir.exists():
        return []

    return [
        {
            "name": file.name,
            "url": f"/api/indic-evaluation/charts/{file.name}"
        }
        for file in sorted(chart_dir.glob("*.png"))
    ]


@app.get("/api/indic-evaluation/charts/{filename}")
def indic_get_chart(filename: str):

    # Only allow PNG filenames, not arbitrary paths
    if Path(filename).name != filename or not filename.lower().endswith(".png"):
        raise HTTPException(400, "Invalid chart filename")

    file_path = INDIC_DATA_DIR / "charts" / filename

    if not file_path.is_file():
        raise HTTPException(404, "Chart not found")

    return FileResponse(file_path, media_type="image/png")


# ==========================================================
# FRONTEND STATIC FILES — KEEP THIS AT THE VERY END
# ==========================================================

from fastapi.staticfiles import StaticFiles

app.mount(
    "/",
    StaticFiles(directory=ROOT / "frontend", html=True),
    name="frontend"
)