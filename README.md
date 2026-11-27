# Image Forensics

A plain HTML/CSS/JS workspace for three checks:

1. **Compare** — align a before/after pair, show changed regions and a mask.
2. **Tamper check** — compression, noise and duplicated-patch clues in one image.
3. **AI origin** — file tags, a local ONNX detector and visual clues, combined into a confidence band.

## Honest limits

- **AI detection is evidence, not proof.** Detectors miss new generators and fail on compressed images. The app reports `Strong / Moderate / Weak / No evidence`, never a bare verdict.
- **Single-image edit checks are clues.** Re-saving an image can look like tampering. ELA is most meaningful for JPEG uploads.
- **No metadata tag means unknown**, not human-made.

## Run locally

```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 7860
```

Open `http://localhost:7860`. Python 3.11+ is recommended. The first AI-origin check downloads a 93 MB quantized model (~10–20 s on CPU); no PyTorch or CUDA is needed.

## Checks

```bash
python test_core.py
```

Ten assertions on synthetic images. No test framework required.

## How it works

| Page | Method |
|---|---|
| Compare | `findTransformECC` alignment → Gaussian blur → absolute difference → threshold → morphological open → contours above an area limit |
| Tamper check | Error Level Analysis (JPEG Q90 re-save), ORB copy-move matching, 16×16 block noise inconsistency |
| AI origin | PNG `parameters`/`workflow` chunks, EXIF Software, XMP/C2PA markers → ONNX Swin detector (`onnxruntime`, int8) → Laplacian smoothness and saturation clipping |

Uploaded images are downscaled to a 1600 px longest edge before processing.

## Deployment (Hugging Face Spaces)

- SDK: Docker
- Port: 7860 (the Dockerfile respects the `PORT` environment variable)
- Free CPU tier works; expect a cold start on the first AI-origin check

## Files

```
server.py            FastAPI endpoints and static file serving
static/index.html    page structure
static/style.css     layout and styling
static/app.js        uploads, previews, API calls and result rendering
core.py              pure image-processing functions
test_core.py         synthetic-image checks
Dockerfile           container for HF Spaces
```

## License

App code: MIT.

The AI detector, [`SMOGY-Ai-images-detector-ONNX`](https://huggingface.co/onnx-community/SMOGY-Ai-images-detector-ONNX), is licensed **CC-BY-NC-4.0** (non-commercial). It is downloaded at runtime and is not covered by this repository's MIT license. Check that license before commercial use.
