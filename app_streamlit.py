import io
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import streamlit as st
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

st.set_page_config(page_title="Image Forensics", page_icon="🔎", layout="wide")
st.title("🔎 Image Forensics")
st.write("Compare edits, inspect a single image for tampering, and assess AI origin.")

CAVEAT = ("Detectors miss new AI generators and fail on compressed images. "
          "Treat every result as evidence, never proof.")


@st.cache_data(show_spinner=False)
def cached_diff(before_bytes, after_bytes, threshold, min_area, align):
    before = Image.open(io.BytesIO(before_bytes)).convert("RGB")
    after = Image.open(io.BytesIO(after_bytes)).convert("RGB")
    return detect_changes(before, after, threshold, min_area, align)


@st.cache_resource(show_spinner="Downloading AI detector (93 MB, first time only)…")
def load_detector():
    import onnxruntime as ort
    target = Path.home() / ".cache" / "image-forensics" / MODEL_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size < 50_000_000:
        urlretrieve(MODEL_URL, target)
    return ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])


compare_tab, tamper_tab, ai_tab = st.tabs([
    "Compare two images", "Check one image", "AI origin"
])

with compare_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Before")
        before_file = st.file_uploader("Before image", type=["jpg", "png", "jpeg"],
                                       key="before")
        if before_file:
            st.image(before_file, width="stretch")
    with right:
        st.subheader("After")
        after_file = st.file_uploader("After image", type=["jpg", "png", "jpeg"],
                                      key="after")
        if after_file:
            st.image(after_file, width="stretch")

    if before_file and after_file:
        controls = st.columns(3)
        threshold = controls[0].slider("Difference threshold", 1, 100, 30)
        min_area = controls[1].slider("Minimum region area", 20, 2000, 100, step=20)
        align = controls[2].checkbox("Align images first", value=True)
        result = cached_diff(before_file.getvalue(), after_file.getvalue(),
                             threshold, min_area, align)
        if align and result.aligned:
            st.caption("Images were aligned automatically before comparison.")
        elif align:
            st.caption("Automatic alignment failed; comparing as uploaded.")

        out_left, out_right = st.columns(2)
        with out_left:
            st.subheader("Marked changes")
            st.image(result.marked, width="stretch")
        with out_right:
            st.subheader("Change mask")
            st.image(result.mask, width="stretch")
            with st.expander("Show raw difference"):
                st.image(result.diff, width="stretch")

        st.metric("Changed regions", len(result.regions))
        if result.regions:
            st.dataframe(result.regions, width="stretch")
        else:
            st.info("No regions passed the area filter.")

with tamper_tab:
    st.write("Clues only: a clean image can look suspicious after re-saving.")
    upload = st.file_uploader("Image to inspect", type=["jpg", "png", "jpeg"],
                              key="tamper")
    if upload:
        image = Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
        if upload.type != "image/jpeg":
            st.caption("ELA is most meaningful for JPEG uploads.")
        st.image(image, width="stretch")

        with st.spinner("Analyzing..."):
            heat, ela_score = error_level_analysis(image)
            pairs = copy_move(image)
            noise, ratio = noise_map(image)

        c1, c2, c3 = st.columns(3)
        c1.metric("ELA mean difference", f"{ela_score:.1f}")
        c1.caption("Average re-compression change; not a tamper score")
        c2.metric("Copy-move pairs", len(pairs))
        c2.caption("Matching patches far apart")
        c3.metric("Noise ratio", f"{ratio:.1f}")
        c3.caption("Max block noise ÷ median block noise")

        h1, h2, h3 = st.columns(3)
        with h1:
            st.subheader("ELA heatmap")
            st.caption("Unusually bright or dark blocks can be edited areas.")
            st.image(heat, width="stretch")
        with h2:
            st.subheader("Noise map")
            st.image(noise, width="stretch", clamp=True)
        with h3:
            st.subheader("Copy-move matches")
            marked = prepare(image)
            for a, b in pairs[:40]:
                cv2.line(marked, a, b, (255, 0, 0), 1)
            st.image(marked, width="stretch")

with ai_tab:
    st.write("Three independent signals, combined into a confidence band.")
    ai_upload = st.file_uploader("Image to assess", type=["jpg", "png", "jpeg"],
                                 key="ai")
    if ai_upload:
        raw = ai_upload.getvalue()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        st.image(image, width="stretch")

        provenance = read_provenance(raw)
        flags, smoothness, clipped = ai_heuristics(image)

        st.subheader("Signals")
        if provenance.hits:
            for hit in provenance.hits:
                st.success(f"File tag — {hit}")
        else:
            st.write("File tags: none found")

        with st.spinner("Running AI detector…"):
            try:
                detector = load_detector()
                p_artificial = artificial_probability(detector, image)
                st.write(f"Detector: {p_artificial:.0%} artificial")
            except Exception as exc:
                p_artificial = None
                st.warning(f"Detector unavailable: {exc}")

        st.write(f"Visual clues: {', '.join(flags) if flags else 'none'}")
        with st.expander("Raw numbers"):
            st.write(f"Texture smoothness (Laplacian variance): {smoothness:.1f}")
            st.write(f"Clipped-saturation fraction: {clipped:.2%}")

        band, reason = ai_band(provenance, p_artificial, flags)
        st.subheader(f"Assessment: {band}")
        st.write(reason)
        st.caption(CAVEAT)

st.caption(CAVEAT)
