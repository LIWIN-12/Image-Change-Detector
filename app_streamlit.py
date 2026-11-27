import html
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

CAVEAT = ("Detectors miss new AI generators and fail on compressed images. "
          "Treat every result as evidence, never proof.")

CSS = """
<style>
.stApp { background: linear-gradient(180deg, #f7f9fc 0%, #edf2f8 100%); }
.block-container { padding-top: 1.5rem; max-width: 1500px; }
.hero {
  background: linear-gradient(135deg, #0f172a, #1e293b 55%, #312e81);
  color: white; padding: 26px 30px; border-radius: 20px;
  box-shadow: 0 12px 30px rgba(15, 23, 42, .18); margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 2.1rem; letter-spacing: -.02em; }
.hero p { margin: 8px 0 14px; color: #cbd5e1; font-size: .98rem; }
.pill-row { display: flex; gap: 8px; flex-wrap: wrap; }
.hero-pill {
  background: rgba(255, 255, 255, .12); border: 1px solid rgba(255, 255, 255, .22);
  padding: 5px 12px; border-radius: 999px; font-size: .82rem;
}
.section-label { font-weight: 700; color: #334155; font-size: 1.02rem; margin: 12px 0 8px; }
.tile {
  background: white; border: 1px solid #e2e8f0; border-radius: 14px;
  padding: 14px 16px; box-shadow: 0 2px 8px rgba(15, 23, 42, .05); height: 100%;
}
.tile-label { color: #64748b; font-size: .76rem; text-transform: uppercase; letter-spacing: .06em; }
.tile-value { font-size: 1.55rem; font-weight: 700; color: #0f172a; margin-top: 4px; }
.tile-note { color: #64748b; font-size: .8rem; margin-top: 4px; }
.band { border-radius: 16px; padding: 20px; color: white; box-shadow: 0 6px 18px rgba(15, 23, 42, .12); }
.band-label { font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; opacity: .85; }
.band-value { font-size: 2rem; font-weight: 800; margin: 4px 0; }
.band-strong { background: linear-gradient(135deg, #b91c1c, #dc2626); }
.band-moderate { background: linear-gradient(135deg, #b45309, #f59e0b); }
.band-weak { background: linear-gradient(135deg, #0369a1, #0ea5e9); }
.band-none { background: linear-gradient(135deg, #047857, #10b981); }
.band-unknown { background: linear-gradient(135deg, #475569, #64748b); }
.chip {
  display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: .8rem;
  margin: 2px 4px 2px 0; background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe;
}
.chip-good { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
.chip-warn { background: #fffbeb; color: #92400e; border-color: #fde68a; }
.note {
  background: white; border-left: 4px solid #4f46e5; border-radius: 10px;
  padding: 12px 16px; color: #475569; font-size: .89rem;
}
.guide { font-size: .87rem; color: #475569; line-height: 1.55; }
[data-testid="stTabs"] button { font-size: 1rem; font-weight: 600; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>🔎 Image Forensics</h1>'
    '<p>Compare edits, inspect tampering clues, and assess AI origin — in one place.</p>'
    '<div class="pill-row"><span class="hero-pill">📁 local files only</span>'
    '<span class="hero-pill">⚡ processed in your session</span>'
    '<span class="hero-pill">🧠 ONNX detector</span></div></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Quick guide")
    st.markdown(
        '<div class="guide">'
        '<b>Compare</b> — add a before/after pair. Images are aligned first, then '
        'differences become boxes and a mask.<br><br>'
        '<b>Check one image</b> — look for compression, noise and duplicated-patch '
        'clues.<br><br>'
        '<b>AI origin</b> — combine file tags, a detector score and visual clues '
        'into a confidence band.'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Read this first"):
        st.write(CAVEAT)
        st.write("A clean image can look suspicious after re-saving. No tag means "
                 "'unknown', never 'human-made'.")


def tile(label, value, note=None):
    body = (f'<div class="tile"><div class="tile-label">{html.escape(str(label))}</div>'
            f'<div class="tile-value">{html.escape(str(value))}</div>')
    if note:
        body += f'<div class="tile-note">{html.escape(str(note))}</div>'
    st.markdown(body + "</div>", unsafe_allow_html=True)


def chip(text, tone=""):
    css_class = f"chip {tone}".strip()
    st.markdown(f'<span class="{css_class}">{html.escape(str(text))}</span>',
                unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="section-label">{html.escape(text)}</div>',
                unsafe_allow_html=True)


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
    "🔁  Compare two images", "🔍  Check one image", "🤖  AI origin"
])

with compare_tab:
    section("1 · Upload the pair")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Before**")
            before_file = st.file_uploader("Before image", type=["jpg", "png", "jpeg"],
                                           key="before")
            if before_file:
                st.image(before_file, width="stretch")
    with right:
        with st.container(border=True):
            st.markdown("**After**")
            after_file = st.file_uploader("After image", type=["jpg", "png", "jpeg"],
                                          key="after")
            if after_file:
                st.image(after_file, width="stretch")

    if before_file and after_file:
        section("2 · Tune detection")
        with st.container(border=True):
            controls = st.columns(3)
            threshold = controls[0].slider("Difference threshold", 1, 100, 30)
            min_area = controls[1].slider("Minimum region area", 20, 2000, 100, step=20)
            align = controls[2].checkbox("Align images first", value=True)

        result = cached_diff(before_file.getvalue(), after_file.getvalue(),
                             threshold, min_area, align)
        if align and result.aligned:
            chip("✓ Images aligned automatically", "chip-good")
        elif align:
            chip("! Alignment failed — compared as uploaded", "chip-warn")

        section("3 · Results")
        metrics = st.columns(4)
        with metrics[0]:
            tile("Changed regions", len(result.regions))
        with metrics[1]:
            tile("Largest region",
                 max((r["area"] for r in result.regions), default=0), "pixels")
        with metrics[2]:
            tile("Mean difference",
                 round(float(result.diff.mean()), 1), "gray levels")
        with metrics[3]:
            tile("Alignment", "Yes" if result.aligned else "No")

        images = st.columns(3)
        with images[0]:
            with st.container(border=True):
                st.markdown("**Marked changes**")
                st.image(result.marked, width="stretch")
        with images[1]:
            with st.container(border=True):
                st.markdown("**Change mask**")
                st.image(result.mask, width="stretch")
        with images[2]:
            with st.container(border=True):
                st.markdown("**Aligned after image**")
                st.image(result.after, width="stretch")

        table_col, action_col = st.columns([4, 1])
        with table_col:
            with st.expander("Region details"):
                if result.regions:
                    st.dataframe(result.regions, width="stretch")
                else:
                    st.info("No regions passed the area filter.")
        with action_col:
            _, encoded = cv2.imencode(
                ".png", cv2.cvtColor(result.marked, cv2.COLOR_RGB2BGR)
            )
            st.download_button("⬇ Save marked image", encoded.tobytes(),
                               file_name="marked.png", mime="image/png")
        with st.expander("Show raw difference"):
            st.image(result.diff, width="stretch")

with tamper_tab:
    st.markdown('<div class="note">Clues only — a clean image can look '
                'suspicious after re-saving.</div>', unsafe_allow_html=True)
    section("Upload one image")
    with st.container(border=True):
        upload = st.file_uploader("Image to inspect", type=["jpg", "png", "jpeg"],
                                  key="tamper")
    if upload:
        image = Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
        with st.spinner("Analyzing..."):
            heat, ela_score = error_level_analysis(image)
            pairs = copy_move(image)
            noise, ratio = noise_map(image)

        if upload.type != "image/jpeg":
            chip("ELA is most meaningful for JPEG uploads", "chip-warn")

        preview, scores = st.columns([1, 2])
        with preview:
            with st.container(border=True):
                st.markdown("**Uploaded image**")
                st.image(image, width="stretch")
        with scores:
            score_cols = st.columns(3)
            with score_cols[0]:
                tile("ELA mean diff", f"{ela_score:.1f}", "not a tamper score")
            with score_cols[1]:
                tile("Copy-move pairs", len(pairs), "matching patches")
            with score_cols[2]:
                tile("Noise ratio", f"{ratio:.1f}", "max ÷ median block")

        section("Visual checks")
        h1, h2, h3 = st.columns(3)
        with h1:
            with st.container(border=True):
                st.markdown("**ELA heatmap**")
                st.caption("Unusually bright or dark blocks can be edited areas.")
                st.image(heat, width="stretch")
        with h2:
            with st.container(border=True):
                st.markdown("**Noise map**")
                st.caption("Blocks with unusual high-frequency detail.")
                st.image(noise, width="stretch", clamp=True)
        with h3:
            with st.container(border=True):
                st.markdown("**Copy-move matches**")
                st.caption("Lines link patches that look duplicated.")
                marked = prepare(image)
                for a, b in pairs[:40]:
                    cv2.line(marked, a, b, (255, 0, 0), 1)
                st.image(marked, width="stretch")

with ai_tab:
    st.markdown('<div class="note">Three independent signals, combined into '
                'a confidence band — never a bare verdict.</div>',
                unsafe_allow_html=True)
    section("Upload one image")
    with st.container(border=True):
        ai_upload = st.file_uploader("Image to assess", type=["jpg", "png", "jpeg"],
                                     key="ai")
    if ai_upload:
        raw = ai_upload.getvalue()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        provenance = read_provenance(raw)
        flags, smoothness, clipped = ai_heuristics(image)

        preview, verdict = st.columns([1, 1])
        with preview:
            with st.container(border=True):
                st.markdown("**Uploaded image**")
                st.image(image, width="stretch")
        with verdict:
            with st.spinner("Running AI detector…"):
                try:
                    detector = load_detector()
                    p_artificial = artificial_probability(detector, image)
                except Exception as exc:
                    p_artificial = None
                    st.warning(f"Detector unavailable: {exc}")
            band, reason = ai_band(provenance, p_artificial, flags)
            tone = {
                "Strong": "strong", "Moderate": "moderate", "Weak": "weak",
                "No evidence": "none", "Unknown": "unknown",
            }[band]
            st.markdown(
                f'<div class="band band-{tone}"><div class="band-label">'
                f'Assessment</div><div class="band-value">{html.escape(band)}</div>'
                f'<div>{html.escape(reason)}</div></div>',
                unsafe_allow_html=True,
            )

        section("Signals")
        signal_cols = st.columns(3)
        with signal_cols[0]:
            with st.container(border=True):
                st.markdown("**File tags**")
                if provenance.hits:
                    for hit in provenance.hits:
                        chip(hit, "chip-warn")
                else:
                    chip("None found", "chip-good")
        with signal_cols[1]:
            with st.container(border=True):
                st.markdown("**AI detector**")
                if p_artificial is not None:
                    tile("Artificial probability", f"{p_artificial:.0%}")
                    st.progress(p_artificial)
                else:
                    tile("Artificial probability", "n/a")
        with signal_cols[2]:
            with st.container(border=True):
                st.markdown("**Visual clues**")
                if flags:
                    for flag in flags:
                        chip(flag, "chip-warn")
                else:
                    chip("None found", "chip-good")

        with st.expander("Raw numbers"):
            st.write(f"Texture smoothness (Laplacian variance): {smoothness:.1f}")
            st.write(f"Clipped-saturation fraction: {clipped:.2%}")
        st.markdown(f'<div class="note">{html.escape(CAVEAT)}</div>',
                    unsafe_allow_html=True)
