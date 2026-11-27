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
:root {
  --paper: #f7f6f2;
  --ink: #17191c;
  --muted: #6d7278;
  --line: #d9d7d0;
  --line-strong: #b9b6ad;
  --accent: #185a48;
}
.stApp { background: var(--paper); }
.block-container { max-width: 1320px; padding-top: 1.25rem; }
.masthead { border-bottom: 2px solid var(--ink); padding-bottom: 14px; margin-bottom: 16px; }
.overline {
  font: 600 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: .14em; text-transform: uppercase; color: var(--muted);
}
.masthead h1 { font-size: 2rem; font-weight: 650; margin: 6px 0 4px; }
.masthead p { margin: 0; color: var(--muted); font-size: .92rem; }
.sec {
  display: flex; align-items: baseline; gap: 10px; font-weight: 600;
  border-top: 1px solid var(--line); padding-top: 10px; margin: 18px 0 10px;
}
.sec .num {
  font: 600 11px ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--accent);
}
.metric { background: white; border: 1px solid var(--line); padding: 12px 14px; }
.metric-label {
  font: 600 10px ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: .1em; text-transform: uppercase; color: var(--muted);
}
.metric-value {
  font: 600 1.45rem ui-monospace, SFMono-Regular, Menlo, monospace;
  font-variant-numeric: tabular-nums; margin-top: 6px;
}
.metric-note { font-size: .78rem; color: var(--muted); margin-top: 4px; }
.tag {
  display: inline-block; font: 500 11px ui-monospace, SFMono-Regular, Menlo, monospace;
  border: 1px solid var(--line-strong); padding: 3px 7px; margin: 2px 4px 2px 0;
  color: var(--ink);
}
.tag-ok { border-color: #9bbcaf; color: var(--accent); }
.tag-warn { border-color: #d6ad9c; color: #a4472b; }
.note { border-left: 3px solid var(--accent); padding: 8px 12px; color: var(--muted); font-size: .86rem; }
.verdict { background: white; border: 1px solid var(--line); border-left: 5px solid var(--accent); padding: 16px; }
.verdict-label {
  font: 600 10px ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: .12em; text-transform: uppercase; color: var(--muted);
}
.verdict-value { font-size: 1.7rem; font-weight: 650; margin: 4px 0; }
.verdict-strong { border-left-color: #a4472b; }
.verdict-moderate { border-left-color: #b07b2a; }
.verdict-weak { border-left-color: #2b6a8f; }
.verdict-none { border-left-color: var(--accent); }
.verdict-unknown { border-left-color: var(--muted); }
.guide { font-size: .86rem; color: #454a50; line-height: 1.6; }
.guide b {
  font: 600 .78rem ui-monospace, SFMono-Regular, Menlo, monospace;
  text-transform: uppercase; letter-spacing: .06em;
}
[data-testid="stTabs"] button { font-size: .92rem; font-weight: 600; }
div[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-radius: 2px !important; border-color: var(--line) !important; box-shadow: none !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="masthead"><div class="overline">Local image forensics</div>'
    '<h1>Image Forensics</h1>'
    '<p>Compare edits, inspect tampering clues, and assess AI origin.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Quick guide")
    st.markdown(
        '<div class="guide">'
        '<b>Compare</b> — add a before/after pair. Images are aligned first, '
        'then differences become boxes and a mask.<br><br>'
        '<b>Check one image</b> — look for compression, noise and '
        'duplicated-patch clues.<br><br>'
        '<b>AI origin</b> — combine file tags, a detector score and visual '
        'clues into a confidence band.'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Read this first"):
        st.write(CAVEAT)
        st.write("A clean image can look suspicious after re-saving. No tag means "
                 "'unknown', never 'human-made'.")


def tile(label, value, note=None):
    body = (f'<div class="metric"><div class="metric-label">{html.escape(str(label))}</div>'
            f'<div class="metric-value">{html.escape(str(value))}</div>')
    if note:
        body += f'<div class="metric-note">{html.escape(str(note))}</div>'
    st.markdown(body + "</div>", unsafe_allow_html=True)


def chip(text, tone=""):
    css_class = f"tag {tone}".strip()
    st.markdown(f'<span class="{css_class}">{html.escape(str(text))}</span>',
                unsafe_allow_html=True)


def section(number, text):
    st.markdown(
        f'<div class="sec"><span class="num">{number}</span>{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


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


compare_tab, tamper_tab, ai_tab = st.tabs(["Compare", "Tamper check", "AI origin"])

with compare_tab:
    section("01", "Upload the pair")
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
        section("02", "Tune detection")
        with st.container(border=True):
            controls = st.columns(3)
            threshold = controls[0].slider("Difference threshold", 1, 100, 30)
            min_area = controls[1].slider("Minimum region area", 20, 2000, 100, step=20)
            align = controls[2].checkbox("Align images first", value=True)

        result = cached_diff(before_file.getvalue(), after_file.getvalue(),
                             threshold, min_area, align)
        if align and result.aligned:
            chip("Aligned automatically", "tag-ok")
        elif align:
            chip("Alignment failed — compared as uploaded", "tag-warn")

        section("03", "Results")
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
            st.download_button("Save marked image", encoded.tobytes(),
                               file_name="marked.png", mime="image/png")
        with st.expander("Show raw difference"):
            st.image(result.diff, width="stretch")

with tamper_tab:
    st.markdown('<div class="note">Clues only — a clean image can look '
                'suspicious after re-saving.</div>', unsafe_allow_html=True)
    section("01", "Upload one image")
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
            chip("ELA is most meaningful for JPEG uploads", "tag-warn")

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

        section("02", "Visual checks")
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
    section("01", "Upload one image")
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
                f'<div class="verdict verdict-{tone}">'
                f'<div class="verdict-label">Assessment</div>'
                f'<div class="verdict-value">{html.escape(band)}</div>'
                f'<div>{html.escape(reason)}</div></div>',
                unsafe_allow_html=True,
            )

        section("02", "Signals")
        signal_cols = st.columns(3)
        with signal_cols[0]:
            with st.container(border=True):
                st.markdown("**File tags**")
                if provenance.hits:
                    for hit in provenance.hits:
                        chip(hit, "tag-warn")
                else:
                    chip("None found", "tag-ok")
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
                        chip(flag, "tag-warn")
                else:
                    chip("None found", "tag-ok")

        with st.expander("Raw numbers"):
            st.write(f"Texture smoothness (Laplacian variance): {smoothness:.1f}")
            st.write(f"Clipped-saturation fraction: {clipped:.2%}")
        st.markdown(f'<div class="note">{html.escape(CAVEAT)}</div>',
                    unsafe_allow_html=True)
