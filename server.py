"""Small HTTP API for the browser UI. All work stays in core.py."""

import base64
import io
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from core import (
    MODEL_FILE,
    MODEL_URL,
    ai_band,
    ai_heuristics,
    artificial_probability,
    copy_move,
    detect_changes,
    error_level_analysis,
    noise_map,
    prepare,
    read_provenance,
)

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Image Forensics")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def png_url(array: np.ndarray) -> str:
    """Encode an RGB or grayscale array as a PNG data URL."""
    ok, buffer = cv2.imencode(".png", array)
    if not ok:
        raise ValueError("Could not encode image")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return "data:image/png;base64," + encoded


def pil_image(upload: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(upload.file.read())).convert("RGB")


@lru_cache(maxsize=1)
def detector():
    import onnxruntime as ort
    from urllib.request import urlretrieve

    target = Path.home() / ".cache" / "image-forensics" / MODEL_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size < 50_000_000:
        urlretrieve(MODEL_URL, target)
    return ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/compare")
async def compare(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    threshold: int = Form(30),
    min_area: int = Form(100),
    align: bool = Form(True),
):
    try:
        result = detect_changes(pil_image(before), pil_image(after),
                                threshold, min_area, align)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return {
        "regions": result.regions,
        "aligned": result.aligned,
        "mean_diff": round(float(result.diff.mean()), 1),
        "marked": png_url(result.marked),
        "mask": png_url(result.mask),
        "after": png_url(result.after),
        "diff": png_url(result.diff),
    }


@app.post("/api/tamper")
async def tamper(image: UploadFile = File(...)):
    try:
        array = prepare(pil_image(image))
        heat, ela_score = error_level_analysis(array)
        pairs = copy_move(array)
        noise, ratio = noise_map(array)
        marked = array.copy()
        for a, b in pairs[:40]:
            cv2.line(marked, a, b, (255, 0, 0), 1)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return {
        "ela": round(float(ela_score), 1),
        "pairs": len(pairs),
        "ratio": round(float(ratio), 1),
        "heat": png_url(heat),
        "noise": png_url(noise.astype(np.uint8)),
        "marked": png_url(marked),
        "is_jpeg": image.content_type == "image/jpeg",
    }


@app.post("/api/ai")
async def ai(image: UploadFile = File(...)):
    try:
        raw = image.file.read()
        array = prepare(Image.open(io.BytesIO(raw)).convert("RGB"))
        provenance = read_provenance(raw)
        flags, smoothness, clipped = ai_heuristics(array)
        probability = artificial_probability(detector(), array)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    band, reason = ai_band(provenance, probability, flags)
    return {
        "band": band,
        "reason": reason,
        "probability": round(probability, 4),
        "hits": provenance.hits,
        "flags": flags,
        "smoothness": round(smoothness, 1),
        "clipped": round(clipped, 4),
    }
