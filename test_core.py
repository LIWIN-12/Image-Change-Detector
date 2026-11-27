"""Small checks for the pure logic. Run: python test_core.py"""

import io

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

from core import (
    ai_band,
    ai_heuristics,
    artificial_probability,
    copy_move,
    detector_input,
    detect_changes,
    error_level_analysis,
    noise_map,
    prepare,
    read_provenance,
)


def textured(size=200, seed=3):
    """Spatially coherent texture; white noise would vanish under the blur step."""
    rng = np.random.default_rng(seed)
    small = rng.integers(30, 200, (size // 8, size // 8, 3), dtype=np.uint8)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)


def test_identical_images_give_no_regions():
    image = textured()
    assert detect_changes(image, image, align=False).regions == []


def test_pasted_square_is_one_region():
    before = textured()
    after = before.copy()
    after[70:130, 80:140] = 250
    regions = detect_changes(before, after, align=False).regions
    assert len(regions) == 1, regions
    x, y, w, h = (regions[0][k] for k in ("x", "y", "w", "h"))
    assert abs(x - 80) < 10 and abs(y - 70) < 10
    assert abs(w - 60) < 10 and abs(h - 60) < 10


def test_alignment_removes_small_shift():
    before = textured(seed=11)
    shift = np.float32([[1, 0, 5], [0, 1, -3]])
    after = cv2.warpAffine(before, shift, (200, 200), borderMode=cv2.BORDER_REFLECT)
    raw = len(detect_changes(before, after, align=False).regions)
    fixed = detect_changes(before, after, align=True)
    assert raw > 0, "shift should be visible without alignment"
    assert fixed.aligned, "ECC should converge on a clean translation"
    assert len(fixed.regions) == 0, fixed.regions


def test_ela_returns_a_finite_heatmap():
    heat, score = error_level_analysis(textured())
    assert heat.shape == prepare(textured()).shape
    assert np.isfinite(score) and score >= 0


def test_copy_move_finds_duplicated_patch():
    image = textured(seed=21)
    image[130:170, 130:170] = image[40:80, 40:80]
    assert len(copy_move(image)) >= 1


def test_noise_map_flags_smooth_patch():
    image = textured(seed=31)
    image[40:80, 40:80] = 128
    _, ratio = noise_map(image)
    assert ratio > 1.5, ratio


def test_provenance_reads_png_parameters():
    buffer = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "Stable Diffusion prompt: a red fox")
    Image.fromarray(textured()).save(buffer, format="PNG", pnginfo=info)
    assert read_provenance(buffer.getvalue()).strong

    clean = io.BytesIO()
    Image.fromarray(textured()).save(clean, format="PNG")
    assert not read_provenance(clean.getvalue()).strong


def test_detector_tensor_shape():
    tensor = detector_input(textured())
    assert tensor.shape == (1, 3, 224, 224)
    assert np.isfinite(tensor).all()


def test_artificial_probability_uses_label_zero():
    class FakeSession:
        def get_inputs(self):
            return [type("Input", (), {"name": "pixel_values"})()]

        def run(self, _, feeds):
            assert feeds["pixel_values"].shape == (1, 3, 224, 224)
            return [np.array([[2.0, -2.0]], dtype=np.float32)]

    probability = artificial_probability(FakeSession(), textured())
    assert probability > 0.95, probability


def test_heuristics_and_bands():
    flags, smoothness, clipped = ai_heuristics(textured())
    assert isinstance(flags, list) and smoothness >= 0 and 0 <= clipped <= 1
    assert ai_band(read_provenance(b""), 0.99, [])[0] == "Moderate"
    assert ai_band(read_provenance(b"c2pa"), 0.1, [])[0] == "Strong"
    assert ai_band(read_provenance(b""), 0.1, [])[0] == "No evidence"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} checks passed")
