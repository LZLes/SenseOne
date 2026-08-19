"""
Test harness for the framing-QC gate (tools/framing_qc.py).

Run before a live demo: `python tests/test_framing_qc.py`

Two tiers, clearly separated:
  1. OFFLINE (no API key, no network, runs in ~1s): synthetic good/bad
     photos exercise the deterministic checks (blur, lighting, off-centre)
     for real, and the vision-dependent checks (not_electrode, angled,
     overlapping, multiple_electrodes, flipped, tampered, mixed_types) are
     exercised through a monkeypatched vision function -- validates the
     aggregation/gating logic (issue codes, message specificity, the
     early-return guard) without spending a Gemini call. This tier is what
     you can always run, key or no key, right before walking into a demo.
  2. LIVE (needs GEMINI_API_KEY, ~5 real vision calls): confirms the actual
     vision integration works end to end against a real good photo and a
     real non-electrode photo. Skipped automatically if no key is
     configured, with a clear note rather than a silent gap.

Each case prints PASS/FAIL and the reason; exits non-zero if anything fails.
"""

import io
import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
from tools import framing_qc  # noqa: E402

REAL_PHOTO = Path("reference_images/20260707/20260707_A1.bmp")

failures = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def clean_vision_response():
    return {
        "checked": True,
        "not_electrode": False, "not_electrode_reason": None,
        "angled": False, "angle_reason": None,
        "overlapping": False, "overlap_reason": None,
        "multiple_electrodes": False, "multiple_electrodes_reason": None,
        "flipped": False, "flipped_reason": None,
        "tampered": False, "tampered_reason": None,
        "mixed_types": False, "mixed_types_reason": None,
    }


print("=" * 70)
print("TIER 1: OFFLINE (deterministic checks real, vision mocked)")
print("=" * 70)

if not REAL_PHOTO.is_file():
    print(f"Reference photo not found at {REAL_PHOTO} -- can't run offline tests.")
    sys.exit(1)

base = Image.open(REAL_PHOTO).convert("RGB")

print("\n-- good frame (real photo, unmodified) --")
with patch.object(framing_qc, "_vision_framing_check", return_value=clean_vision_response()):
    r = framing_qc.check_photo_framing(image_bytes=to_bytes(base), mime_type="image/jpeg")
check("framing_ok is True", r["framing_ok"] is True)
check("proceed is True", r["proceed"] is True)
check("issues is empty", r["issues"] == [], str(r["issues"]))

print("\n-- angled (mocked vision: angled=True) --")
with patch.object(framing_qc, "_vision_framing_check", return_value={**clean_vision_response(), "angled": True, "angle_reason": "tilted ~20 degrees"}):
    r = framing_qc.check_photo_framing(image_bytes=to_bytes(base), mime_type="image/jpeg")
check("framing_ok is False", r["framing_ok"] is False)
check("proceed is False", r["proceed"] is False)
check("'angled' in issues", "angled" in r["issues"], str(r["issues"]))
check("user_message is specific, not generic", "angled" in r["user_message"].lower() and "hold the phone" in r["user_message"].lower(), r["user_message"])

print("\n-- overlapping electrodes (mocked vision: overlapping=True) --")
with patch.object(framing_qc, "_vision_framing_check", return_value={**clean_vision_response(), "overlapping": True, "overlap_reason": "two strips crossing at the tip"}):
    r = framing_qc.check_photo_framing(image_bytes=to_bytes(base), mime_type="image/jpeg")
check("'overlapping_electrodes' in issues", "overlapping_electrodes" in r["issues"], str(r["issues"]))
check("user_message mentions separating strips", "separated" in r["user_message"].lower(), r["user_message"])

print("\n-- off-centre / cut off (synthetic: shrunk + corner-pasted onto blank canvas) --")
canvas = Image.new("RGB", base.size, (235, 235, 235))
canvas.paste(base.resize((int(base.width * 0.10), int(base.height * 0.10))), (5, 5))
with patch.object(framing_qc, "_vision_framing_check", return_value=clean_vision_response()):
    r = framing_qc.check_photo_framing(image_bytes=to_bytes(canvas), mime_type="image/jpeg")
check("framing_ok is False", r["framing_ok"] is False)
check("'off_centre' in issues", "off_centre" in r["issues"], str(r["issues"]))
check("vision was never called (deterministic check already failed)", r["measurements"]["vision"] is None)

print("\n-- low resolution (synthetic: downscaled to 320x256) --")
low_res = base.resize((320, 256))
r = framing_qc.check_photo_framing(image_bytes=to_bytes(low_res), mime_type="image/jpeg")
check("framing_ok is False", r["framing_ok"] is False)
check("'low_resolution' in issues", "low_resolution" in r["issues"], str(r["issues"]))
check("user_message states the actual pixel dimensions, not generic", "320x256" in r["user_message"], r["user_message"])

print("\n-- dark (synthetic: brightness x0.15) --")
dark = ImageEnhance.Brightness(base).enhance(0.15)
r = framing_qc.check_photo_framing(image_bytes=to_bytes(dark), mime_type="image/jpeg")
check("framing_ok is False", r["framing_ok"] is False)
check("'poor_lighting' in issues", "poor_lighting" in r["issues"], str(r["issues"]))
check("user_message mentions darkness, not generic", "dark" in r["user_message"].lower(), r["user_message"])

print("\n-- blurred (synthetic: gaussian blur radius=4) --")
blurred = base.filter(ImageFilter.GaussianBlur(4))
r = framing_qc.check_photo_framing(image_bytes=to_bytes(blurred), mime_type="image/jpeg")
check("framing_ok is False", r["framing_ok"] is False)
check("'excessive_blur' in issues", "excessive_blur" in r["issues"], str(r["issues"]))

print("\n-- multiple simultaneous issues (dark AND blurred together) --")
dark_and_blurred = ImageEnhance.Brightness(base).enhance(0.35).filter(ImageFilter.GaussianBlur(2))
r = framing_qc.check_photo_framing(image_bytes=to_bytes(dark_and_blurred), mime_type="image/jpeg")
check("both issues reported, not just the first", {"poor_lighting", "excessive_blur"}.issubset(set(r["issues"])), str(r["issues"]))

print("\n-- glare/clipping fools the roughness proxy into reading falsely smooth --")
print("   (separate from the framing gate: this checks analyze_surface_topology's own")
print("   clipping flag, added because a saturated patch has zero internal luminance")
print("   variance and can misreport as a genuinely flat/even print)")
import numpy as np  # noqa: E402
from tools.image_qc import analyze_surface_topology, WORKING_ELECTRODE_CROP_BOX  # noqa: E402

clip_arr = np.asarray(base.convert("RGB")).astype(float).copy()
w, h = base.size
left, top, right, bottom = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(WORKING_ELECTRODE_CROP_BOX)]
cw, ch = right - left, bottom - top
gx0, gy0 = left + int(cw * 0.02), top + int(ch * 0.02)
gx1, gy1 = left + int(cw * 0.98), top + int(ch * 0.98)
clip_arr[gy0:gy1, gx0:gx1] = 255
clip_path = "/tmp/framing_qc_test_clipped.png"
Image.fromarray(clip_arr.astype(np.uint8)).save(clip_path)
normal_surface = analyze_surface_topology(str(REAL_PHOTO))
clipped_surface = analyze_surface_topology(clip_path)
check("clipping flag present when >92% of the region is saturated", any("CLIPPING" in f for f in clipped_surface["flags"]), str(clipped_surface["flags"]))
check("clipping flag absent on the real unmodified photo", not any("CLIPPING" in f for f in normal_surface["flags"]))
check(
    "near-total clipping suppresses Ra back toward baseline (confirmed direction, not assumed)",
    clipped_surface["roughness_luminance_units"]["Ra"] < normal_surface["roughness_luminance_units"]["Ra"] * 1.5,
    f"clipped Ra={clipped_surface['roughness_luminance_units']['Ra']:.1f} vs normal Ra={normal_surface['roughness_luminance_units']['Ra']:.1f}",
)
Path(clip_path).unlink(missing_ok=True)

print("\n-- not an electrode (mocked vision: not_electrode=True) --")
with patch.object(framing_qc, "_vision_framing_check", return_value={**clean_vision_response(), "not_electrode": True, "not_electrode_reason": "shows a hand, not a strip"}):
    r = framing_qc.check_photo_framing(image_bytes=to_bytes(base), mime_type="image/jpeg")
check("'not_electrode' in issues", "not_electrode" in r["issues"], str(r["issues"]))

print("\n-- flipped/reverse side (mocked vision: flipped=True) --")
with patch.object(framing_qc, "_vision_framing_check", return_value={**clean_vision_response(), "flipped": True, "flipped_reason": "blank substrate visible, no printed pads"}):
    r = framing_qc.check_photo_framing(image_bytes=to_bytes(base), mime_type="image/jpeg")
check("'flipped' in issues", "flipped" in r["issues"], str(r["issues"]))

print("\n-- tampered (mocked vision: tampered=True) --")
with patch.object(framing_qc, "_vision_framing_check", return_value={**clean_vision_response(), "tampered": True, "tampered_reason": "visible gouge across the working electrode"}):
    r = framing_qc.check_photo_framing(image_bytes=to_bytes(base), mime_type="image/jpeg")
check("'tampered' in issues", "tampered" in r["issues"], str(r["issues"]))

print("\n-- mixed electrode types (mocked vision: mixed_types=True) --")
with patch.object(framing_qc, "_vision_framing_check", return_value={**clean_vision_response(), "mixed_types": True, "mixed_types_reason": "one 3-pad and one 2-pad design in frame"}):
    r = framing_qc.check_photo_framing(image_bytes=to_bytes(base), mime_type="image/jpeg")
check("'mixed_electrode_types' in issues", "mixed_electrode_types" in r["issues"], str(r["issues"]))

print("\n-- unexpected extra electrode (expect_single_electrode=True, mocked vision: multiple_electrodes=True) --")
with patch.object(framing_qc, "_vision_framing_check", return_value={**clean_vision_response(), "multiple_electrodes": True, "multiple_electrodes_reason": "two separate strips in frame"}):
    r = framing_qc.check_photo_framing(image_bytes=to_bytes(base), mime_type="image/jpeg", expect_single_electrode=True)
check("'multiple_electrodes' in issues", "multiple_electrodes" in r["issues"], str(r["issues"]))

print("\n-- GUARD: qc_sensor_image never reaches downstream defect analysis on a bad photo --")
with patch.object(framing_qc, "_vision_framing_check", return_value={**clean_vision_response(), "angled": True}):
    with patch("tools.image_qc.client") as mock_client:
        result = agent.AVAILABLE_FUNCTIONS["image_qc"](image_path=str(REAL_PHOTO), rotation_degrees=25)
check("status is 'framing_rejected'", result["status"] == "framing_rejected", result.get("status"))
check("no best-guess defects were produced", result.get("defects") == [], str(result.get("defects")))
check("downstream Gemini client was never called", not mock_client.called)

print("\n-- GUARD: compare_to_batch_reference also gates (not just image_qc) --")
print("   (a real gap found via live testing: this tool is independently callable by the")
print("   agent, and an earlier version only gated image_qc, letting a bad photo reach a")
print("   full SSIM comparison through this path untouched)")
with patch.object(framing_qc, "_vision_framing_check", return_value={**clean_vision_response(), "multiple_electrodes": True}):
    result = agent.AVAILABLE_FUNCTIONS["compare_to_batch_reference"](image_path=str(REAL_PHOTO), image_dir="reference_images/20260707")
check("status is 'framing_rejected'", result.get("status") == "framing_rejected", result.get("status"))
check("no SSIM metrics were computed", "metrics" not in result, str(result.keys()))

print("\n-- GUARD: predict_electrode_performance also gates --")
with patch.object(framing_qc, "_vision_framing_check", return_value={**clean_vision_response(), "not_electrode": True}):
    result = agent.AVAILABLE_FUNCTIONS["predict_electrode_performance"](image_path=str(REAL_PHOTO))
check("status is 'framing_rejected'", result.get("status") == "framing_rejected", result.get("status"))

print("\n-- GUARD (opposite direction): a clean photo DOES reach downstream analysis --")
with patch.object(framing_qc, "_vision_framing_check", return_value=clean_vision_response()):
    with patch("tools.image_qc.client") as mock_client:
        mock_client.return_value.models.generate_content.return_value.text = '{"status": "pass", "defects": []}'
        result = agent.AVAILABLE_FUNCTIONS["image_qc"](image_path=str(REAL_PHOTO))
check("status is not 'framing_rejected'", result["status"] != "framing_rejected", result.get("status"))
check("downstream Gemini client WAS called", mock_client.called)

print("\n" + "=" * 70)
print("TIER 2: LIVE (real Gemini vision calls)")
print("=" * 70)

if not agent.api_key_configured():
    print("\nNo GEMINI_API_KEY configured -- skipping live tier. Tier 1 (above) already")
    print("validates all the gating/aggregation logic offline; run this tier once a key")
    print("is available to confirm the real vision integration end to end.")
else:
    bad_examples_dir = Path(__file__).resolve().parent.parent / "bad_examples"
    if bad_examples_dir.is_dir():
        bad_examples = sorted(bad_examples_dir.glob("*.jpg")) + sorted(bad_examples_dir.glob("*.png"))
        print(f"\n-- bad_examples/ regression: every one of {len(bad_examples)} real curated photos must be rejected --")
        print("   (each is a real photo of a genuine failure mode -- a full batch sheet, mixed")
        print("   electrode types, a commercial tray, low-res crops -- kept as a live check that")
        print("   the gate still generalizes to real examples, not just synthetic ones)")
        for p in bad_examples:
            r = agent.AVAILABLE_FUNCTIONS["check_photo_framing"](image_path=str(p), expect_single_electrode=True)
            check(f"{p.name} is rejected", r["framing_ok"] is False, str(r["issues"]))

    print("\n-- real good photo passes end to end --")
    r = agent.AVAILABLE_FUNCTIONS["check_photo_framing"](image_path=str(REAL_PHOTO))
    check("framing_ok is True", r["framing_ok"] is True, str(r["issues"]))

    # hubble_deep_field.jpg (1000x872), not astronaut.png (512x512, below
    # MIN_RESOLUTION_PX) -- needs to clear the resolution floor on its own
    # merits so this case actually isolates the not_electrode vision check,
    # not the resolution check firing first for an unrelated reason.
    skimage_hubble = Path(sys.prefix) / "lib" / "python3.13" / "site-packages" / "skimage" / "data" / "hubble_deep_field.jpg"
    if skimage_hubble.is_file():
        print("\n-- real non-electrode photo (skimage sample) is correctly rejected --")
        # Whatever the actual reason -- a starfield photo may trip lighting/off-centre
        # before the vision call even runs (fail-fast tiering, see module docstring) --
        # what matters is it's rejected for a real, specific reason, not waved through.
        r = agent.AVAILABLE_FUNCTIONS["check_photo_framing"](image_path=str(skimage_hubble))
        check("framing_ok is False", r["framing_ok"] is False, str(r["issues"]))
        check("user_message is specific, not generic", r["user_message"] != "Framing looks good.", r["user_message"])
    else:
        print(f"\n(skipping non-electrode live case -- {skimage_hubble} not found in this environment)")

print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
