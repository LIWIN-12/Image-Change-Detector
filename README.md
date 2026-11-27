# 🔎 Image Forensics

A small Streamlit app for three questions:

1. **What changed between two images?** — align, diff, and list changed regions.
2. **Does one image look edited?** — compression, noise, and duplicated-patch clues.
3. **Does one image look AI-generated?** — file tags, a classifier, and weak visual clues.

## Honest limits

- **AI detection is evidence, not proof.** Detectors miss new generators and fail on compressed or resized images. The app reports `Strong / Moderate / Weak / No evidence`, never a bare verdict.
- **Single-image edit checks are clues.** Re-saving an image can look like tampering. ELA is only meaningful for JPEG uploads, and pasted areas can be darker *or* brighter than their surroundings.
- **Metadata helps only when present.** No tag means "unknown", not "human-made".

## Quick start

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

Python 3.11 or newer is recommended. The first AI-origin check downloads a 93 MB quantized model and takes ~10–20 s on CPU; later checks in the same session are instant. No PyTorch or CUDA install is needed.

## Checks

```bash
python test_core.py
```

Ten assertions on synthetic images. No test framework required.

## How it works

| Tab | Method |
|---|---|
| Compare two images | `findTransformECC` alignment → Gaussian blur → absolute difference → threshold → morphological open → contours above an area limit |
| Check one image | Error Level Analysis (JPEG Q90 re-save), ORB copy-move matching, 16×16 block noise inconsistency |
| AI origin | PNG `parameters`/`workflow` chunks, EXIF Software, XMP/C2PA markers → ONNX Swin detector (`onnxruntime`, int8) → Laplacian smoothness and saturation clipping |

All uploaded images are downscaled to a 1600 px longest edge before processing.

## Deployment (Hugging Face Spaces)

- SDK: Streamlit
- App file: `app_streamlit.py`
- Python: set to 3.11+ in Space settings (do not add `runtime.txt`)
- Free CPU tier works; expect a cold start on first AI-origin check

## Files

```
app_streamlit.py   Streamlit UI, three tabs
core.py            pure image-processing functions
test_core.py       synthetic-image checks
requirements.txt   dependencies
```

## License

App code: MIT.

The AI detector, [`SMOGY-Ai-images-detector-ONNX`](https://huggingface.co/onnx-community/SMOGY-Ai-images-detector-ONNX), is licensed **CC-BY-NC-4.0** (non-commercial). It is downloaded at runtime and is not covered by this repository's MIT license. Check that license before commercial use.
