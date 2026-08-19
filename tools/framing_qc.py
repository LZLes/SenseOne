"""
Framing-QC gate: checks whether a photo is even usable *before* the real
analysis pipeline (qc_sensor_image) spends a Gemini vision call describing
defects in it. A blurry, dark, or badly-cropped photo can't produce a
trustworthy defect read no matter how good the downstream analysis is --
this catches that upstream and refuses to guess, rather than quietly
returning a low-confidence "pass"/"fail" built on unusable input.

Hybrid design (deliberately, not for lack of trying a single approach):
  - Blur, lighting, and off-centre/cut-off are checked with deterministic
    image processing -- free, instant, zero API dependency, and fully
    testable offline against synthetic images.
  - Everything else -- wrong subject entirely (not_electrode), angle,
    overlapping electrodes, unexpected multiple electrodes, flipped/reverse
    side, and physical tampering -- is checked with a single Gemini vision
    call. All of these genuinely need holistic scene understanding a
    geometric heuristic can't reliably provide here (see below), so they're
    deliberately not force-fit into heuristics just to avoid the API call;
    bundled into one call rather than one call per issue to keep the cost
    profile the same as just checking angle/overlap alone.
  - The vision call only runs if the deterministic checks already passed.
    A dark or blurry photo never reaches the API at all -- saves quota on
    the "obviously bad" case, at a disclosed cost: if a photo is BOTH
    badly lit AND, say, angled, only the lighting issue is reported (the
    deterministic checks themselves never short-circuit each other --
    blur/lighting/off-centre are always all three checked and all
    genuine simultaneous issues among *those* are reported together).

Why angle/overlap aren't deterministic here, concretely: a geometric
heuristic (Hough-line angle, contour-overlap) needs to already know what
"normal" looks like for a given rig, and this lab's own reference photos
don't agree with each other on that -- confirmed empirically before writing
any thresholds (see off-centre calibration below). Multiple electrodes
sharing one frame is also completely normal here (a single sheet photo
routinely holds 3 physical electrodes, see sub_position in image_qc.py) --
"overlapping" specifically means electrodes touching/crossing each other,
not "more than one electrode visible," a distinction a vision call can be
told about directly and a pixel-overlap heuristic can't cheaply express.

Off-centre/cut-off calibration note: an early "does the foreground touch
the image border" heuristic was tested against all 189 real photos across
3 batches (20260707/20260804/20260805) before being written down as a
threshold -- it failed immediately. This lab's actual "good" photos
routinely touch the frame edge by design (a wide multi-electrode row
spanning near-full frame width in 20260707; a contact tail touching the
bottom edge consistently across every single 20260804/20260805 photo).
A naive border-touch check would have flagged nearly every real photo.
What ended up robust instead: total foreground *occupancy* (how much of
the frame is electrode content at all) has a floor even though its
absolute value varies a lot batch-to-batch (2.4-31% across the 3
batches) -- so the threshold here is set far below the lowest real floor
observed, catching only a frame that's almost entirely empty/wrong
subject, not fine-grained centering. This is a real, disclosed
limitation: this check catches severe cases, not subtle miscentering.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening, label
from scipy.ndimage import laplace as nd_laplace
from skimage.filters import threshold_otsu

from tools.image_qc import _MIME_TYPES, _validate_image_file

CANONICAL_MAX_DIM = 1280  # metrics below are calibrated at this scale; resizing first keeps
                           # thresholds meaningful regardless of the uploaded photo's native resolution

# Calibrated against all 189 real photos across 3 batches (20260707/20260804/20260805) --
# see module docstring. All thresholds sit with wide margin below/above the *full-dataset*
# observed range, not just a handful of spot-checked examples (an earlier pass calibrated
# against a small sample per batch and missed a real subset of 20260804 with legitimate
# silver-contact glare up to 38.9% saturated pixels -- see fraction_saturated_pixels below).
BLUR_LAPLACIAN_VAR_MIN = 60.0     # real floor observed: 214.9 (softest real photo); even barely-
                                   # perceptible synthetic blur (radius=1) measured at 19.4
DARK_MEAN_LUM_MAX = 90.0          # real floor observed: 150.2
BRIGHT_MEAN_LUM_MIN = 220.0       # real ceiling observed: 174.2
OFF_CENTRE_MIN_OCCUPANCY = 0.015  # real floor observed: 0.0234 (20260707 batch)
OFF_CENTRE_ALL_EDGES_MARGIN = 0.01  # "touches all 4 edges at once" never observed in real data

# The real dataset is shot at a uniform 1280x1024 by one camera rig, so there's no
# real "too low-res" example to calibrate a floor against the way the checks above
# were. This is instead a deliberately generous floor -- well below what any modern
# phone camera produces, but high enough that a thumbnail-sized or heavily
# downscaled upload (which can't support a meaningful defect/roughness read
# regardless of anything else checked here) gets caught rather than silently
# analyzed at a resolution too coarse to say anything real.
MIN_RESOLUTION_PX = (640, 480)


def _check_resolution(image_bytes: bytes) -> dict:
    with Image.open(io.BytesIO(image_bytes)) as img:
        width, height = img.size
    min_w, min_h = MIN_RESOLUTION_PX
    return {"ok": width >= min_w and height >= min_h, "width": width, "height": height}


def _canonical_grayscale(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("L")
        scale = CANONICAL_MAX_DIM / max(img.size)
        if scale < 1:
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BICUBIC)
        return np.asarray(img, dtype=float)


def _check_blur(gray: np.ndarray) -> dict:
    lap_var = float(nd_laplace(gray).var())
    return {"ok": lap_var >= BLUR_LAPLACIAN_VAR_MIN, "laplacian_variance": lap_var}


def _check_lighting(gray: np.ndarray) -> dict:
    """Gates on overall mean luminance alone. fraction_saturated_pixels is
    still measured and reported, but NOT used as a trigger -- calibration
    against the full real dataset found up to 38.9% saturated pixels in
    genuinely good photos (silver-ink contact pads are highly reflective by
    nature, not a lighting defect), while mean luminance stayed tight
    (150.2-174.2) across every real photo regardless of contact glare --
    a far more reliable signal for this dataset than saturated-pixel count.
    """
    mean_lum = float(gray.mean())
    frac_saturated = float((gray > 250).mean())
    too_dark = mean_lum < DARK_MEAN_LUM_MAX
    too_bright = mean_lum > BRIGHT_MEAN_LUM_MIN
    return {
        "ok": not (too_dark or too_bright),
        "too_dark": too_dark, "too_bright": too_bright,
        "mean_luminance": mean_lum, "fraction_saturated_pixels": frac_saturated,
    }


def _largest_foreground_component(gray: np.ndarray):
    """Otsu-threshold, then keep only the largest connected component --
    a raw thresholded mask alone picks up shadows/noise/border artifacts
    scattered around the frame, which corrupted the first version of this
    check's margin measurements. Tries both polarities (electrode vs.
    background can be the darker region depending on lighting/substrate)
    and keeps whichever yields a plausible-sized foreground rather than
    the whole frame or near-nothing -- validated against all 189 real
    calibration photos at zero false positives; a "just take whichever
    side is the minority" version was tried instead and is *simpler*, but
    empirically broke on several real photos near a ~50/50 dark/light
    split, so this asymmetric window (biased toward accepting the darker
    region, since that's what the electrode print itself was in every
    real photo) is deliberately not the more "elegant" version.
    """
    try:
        t = threshold_otsu(gray)
    except ValueError:
        return None
    mask = gray < t
    frac = mask.mean()
    if not (0.02 < frac < 0.7):
        mask = ~mask
    mask = binary_opening(mask, structure=np.ones((5, 5)))
    labeled, n = label(mask)
    if n == 0:
        return None
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    biggest = int(sizes.argmax())
    if sizes[biggest] == 0:
        return None
    return labeled == biggest


def _check_off_centre(gray: np.ndarray) -> dict:
    h, w = gray.shape
    component = _largest_foreground_component(gray)
    if component is None:
        return {"ok": False, "occupancy": 0.0, "reason": "no distinguishable electrode content found in frame"}
    occupancy = float(component.mean())
    ys, xs = np.where(component)
    margins = {
        "left": float(xs.min() / w), "right": float(1 - (xs.max() + 1) / w),
        "top": float(ys.min() / h), "bottom": float(1 - (ys.max() + 1) / h),
    }
    all_edges_touched = all(m < OFF_CENTRE_ALL_EDGES_MARGIN for m in margins.values())
    too_little_content = occupancy < OFF_CENTRE_MIN_OCCUPANCY
    return {
        "ok": not (all_edges_touched or too_little_content),
        "occupancy": occupancy, "margins": margins,
        "all_edges_touched": all_edges_touched, "too_little_content": too_little_content,
    }


_VISION_PROMPT_TEMPLATE = """Look at this photo for framing/subject issues only -- ignore blur,
lighting, and cropping, those are already checked separately. Also ignore print quality
defects (missing ink, streaking, contamination) -- that's a separate downstream analysis,
not this check.

1. NOT_ELECTRODE: does this photo show a screen-printed electrode (SPE) biosensor strip at
   all? Flag this if it's clearly something else entirely (a random object, a person, a blank
   surface, an unrelated lab item) -- not for a real electrode that's just hard to see well.

2. ANGLE: is the electrode strip itself rotated/tilted relative to the frame edges (not
   perfectly axis-aligned)? A few degrees of tilt is fine and common; only flag a clearly,
   visibly angled strip that would make a fixed analysis region miss the electrode.

3. OVERLAP: are two or more physical electrode strips touching or crossing each other?
   IMPORTANT: it is completely normal and expected for ONE photo to contain multiple separate
   electrodes side by side on the same sheet (e.g. a row or grid of several strips) -- that is
   NOT overlap and must NOT be flagged. Only flag this if strips are physically touching,
   crossing, or stacked on top of each other in a way that makes them hard to tell apart.
{multiple_electrodes_instruction}
4. FLIPPED: is the strip shown from its back/reverse side rather than its printed front (e.g.
   you see plain blank substrate instead of the printed working/counter/reference electrode
   pads, or the contact pins are on the visibly wrong side)? Only flag a clear reverse-side
   shot, not just an unusual viewing angle of the correct (front) side.

5. TAMPERED: does the electrode show clear signs of deliberate physical alteration or damage
   -- a cut, hole, scratch gouge, foreign object deliberately placed on it, or similar --
   distinct from a normal manufacturing print defect (which is NOT this check's concern)?
   Only flag obvious physical tampering/damage to the strip itself, not print-quality issues.

6. MIXED_TYPES: if more than one electrode is visible in frame, do they all appear to be the
   SAME electrode type/design (same pad shape and layout, same number of electrodes per
   strip)? Flag mixed_types only if you can see clearly different electrode designs mixed
   together in one frame -- not for minor print-to-print variation of the same design, and
   not applicable at all when only one electrode is visible (leave false).

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"not_electrode": true or false, "not_electrode_reason": "short specific description or null",
 "angled": true or false, "angle_reason": "short specific description or null",
 "overlapping": true or false, "overlap_reason": "short specific description or null",
 "multiple_electrodes": true or false, "multiple_electrodes_reason": "short specific description or null",
 "flipped": true or false, "flipped_reason": "short specific description or null",
 "tampered": true or false, "tampered_reason": "short specific description or null",
 "mixed_types": true or false, "mixed_types_reason": "short specific description or null"}}
"""

_MULTIPLE_ELECTRODES_EXPECTED = """
   Note: this specific photo is EXPECTED to show multiple electrodes on one sheet (a
   catalogued grid photo) -- do not flag multiple_electrodes for that; it's normal here.
"""
_MULTIPLE_ELECTRODES_UNEXPECTED = """3b. MULTIPLE_ELECTRODES: this photo is a researcher's direct upload meant to show ONE
   electrode. Flag multiple_electrodes if the frame actually contains more than one
   separate, distinguishable electrode strip -- that's ambiguous for a single-electrode
   analysis, unlike a catalogued grid photo where it's expected.
"""


def _vision_framing_check(image_bytes: bytes, mime_type: str, expect_single_electrode: bool = False) -> dict:
    """Isolated so tests can monkeypatch this one function and exercise the
    rest of check_photo_framing's aggregation/gating logic without a live
    API call -- see the test harness for why (these checks genuinely need a
    vision call; the rest of this module is deliberately offline-testable).
    """
    import json as _json

    from google.genai import types as _types

    from tools._gemini import client, MODEL, request_slot

    prompt = _VISION_PROMPT_TEMPLATE.format(
        multiple_electrodes_instruction=_MULTIPLE_ELECTRODES_UNEXPECTED if expect_single_electrode else _MULTIPLE_ELECTRODES_EXPECTED
    )
    try:
        with request_slot():
            response = client().models.generate_content(
                model=MODEL,
                contents=[_types.Content(role="user", parts=[
                    _types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    _types.Part.from_text(text=prompt),
                ])],
                config=_types.GenerateContentConfig(temperature=0, thinking_config=_types.ThinkingConfig(include_thoughts=False)),
            )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):]
        parsed = _json.loads(text)
        return {
            "checked": True,
            "not_electrode": bool(parsed.get("not_electrode")), "not_electrode_reason": parsed.get("not_electrode_reason"),
            "angled": bool(parsed.get("angled")), "angle_reason": parsed.get("angle_reason"),
            "overlapping": bool(parsed.get("overlapping")), "overlap_reason": parsed.get("overlap_reason"),
            "multiple_electrodes": bool(parsed.get("multiple_electrodes")) if expect_single_electrode else False,
            "multiple_electrodes_reason": parsed.get("multiple_electrodes_reason"),
            "flipped": bool(parsed.get("flipped")), "flipped_reason": parsed.get("flipped_reason"),
            "tampered": bool(parsed.get("tampered")), "tampered_reason": parsed.get("tampered_reason"),
            "mixed_types": bool(parsed.get("mixed_types")), "mixed_types_reason": parsed.get("mixed_types_reason"),
        }
    except Exception as e:
        # A vision-check failure (network, quota, unparsable response) shouldn't
        # crash the whole framing gate or silently wave the photo through --
        # surface it as its own issue so the caller knows angle/overlap weren't
        # actually verified, rather than defaulting to a false "all clear".
        return {"checked": False, "error": str(e)}


FRAMING_QC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_photo_framing",
        "description": (
            "Check whether an electrode photo is well-framed enough to analyze at all, "
            "BEFORE running sensor_qc-style defect analysis on it -- catches: the subject not "
            "being a recognizable electrode at all, an angled/rotated shot, overlapping "
            "electrodes (or multiple electrodes when only one was expected), mixed electrode "
            "types in one frame, a flipped/reverse-side shot, visible tampering/physical damage, "
            "an off-centre/cut-off frame, poor lighting (too dark or blown-out glare), too-low "
            "resolution, and excessive blur. qc_sensor_image already "
            "runs this automatically as its first step and will refuse to guess on a bad photo, "
            "so you don't need to call this separately before qc_sensor_image -- use it "
            "standalone only when the researcher asks specifically whether a photo is good "
            "enough to use, without wanting a full defect analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to the photo to check."},
                "expect_single_electrode": {
                    "type": "boolean",
                    "description": (
                        "True if this photo is meant to show exactly one electrode (e.g. a "
                        "researcher's fresh upload) -- more than one distinguishable electrode in "
                        "frame is then flagged as ambiguous. False (default) for a catalogued grid "
                        "photo where several electrodes sharing one frame is normal and expected."
                    ),
                },
            },
            "required": ["image_path"],
        },
    },
}


def check_photo_framing(
    image_path: str = "", image_bytes: bytes = None, mime_type: str = "image/jpeg",
    expect_single_electrode: bool = False,
) -> dict:
    """image_path alone (the standalone-tool path): validates and reads the
    file directly. image_bytes+mime_type (the internal qc_sensor_image gate
    path): analyzes exactly those bytes instead -- lets the gate run on an
    already-rotated in-memory image when the caller applied a manual
    rotation_degrees correction, rather than re-flagging the same tilt the
    caller just told it how to fix. image_path is still used for the initial
    file-validity check either way when given.
    """
    if image_path:
        error = _validate_image_file(image_path)
        if error:
            return {"framing_ok": False, "issues": ["unreadable_image"], "user_message": error, "proceed": False}
    if image_bytes is None:
        if not image_path:
            return {"framing_ok": False, "issues": ["unreadable_image"], "user_message": "No image provided.", "proceed": False}
        image_bytes = Path(image_path).read_bytes()
        mime_type = _MIME_TYPES.get(Path(image_path).suffix.lower(), "image/jpeg")

    try:
        gray = _canonical_grayscale(image_bytes)
    except Exception as e:
        return {"framing_ok": False, "issues": ["unreadable_image"], "user_message": f"Could not decode image: {e}", "proceed": False}

    resolution = _check_resolution(image_bytes)
    blur = _check_blur(gray)
    lighting = _check_lighting(gray)
    off_centre = _check_off_centre(gray)

    issues, messages = [], []
    if not resolution["ok"]:
        issues.append("low_resolution")
        min_w, min_h = MIN_RESOLUTION_PX
        messages.append(f"Photo resolution is only {resolution['width']}x{resolution['height']}px (need at least {min_w}x{min_h}px) -- retake at full camera resolution rather than a downscaled/thumbnail export.")
    if not blur["ok"]:
        issues.append("excessive_blur")
        messages.append("Photo is out of focus/blurred -- hold the camera steady and let it focus before capturing, or move closer to a stable surface.")
    if lighting["too_dark"]:
        issues.append("poor_lighting")
        messages.append("Photo is too dark to assess -- retake with more, even lighting.")
    elif lighting["too_bright"]:
        issues.append("poor_lighting")
        messages.append("Photo has blown-out glare on the electrode surface -- reduce direct light/flash or reposition to avoid reflections.")
    if not off_centre["ok"]:
        issues.append("off_centre")
        messages.append("The electrode's working area appears to be cut off or barely visible in frame -- center the strip fully in the shot with clear margin on all sides.")

    vision = None
    # Deliberately skip the vision call once a cheap check has already failed --
    # see module docstring for the cost/completeness tradeoff this accepts.
    if not issues:
        vision = _vision_framing_check(image_bytes, mime_type, expect_single_electrode)
        if not vision.get("checked"):
            issues.append("framing_check_incomplete")
            messages.append(f"Couldn't verify framing/subject ({vision.get('error', 'vision check failed')}) -- try again before relying on this analysis.")
        else:
            # not_electrode first -- if the subject itself is wrong, that's the
            # one issue actually worth leading with; the others are moot on a
            # photo of the wrong thing entirely.
            if vision["not_electrode"]:
                issues.append("not_electrode")
                reason = vision.get("not_electrode_reason") or "this doesn't look like an electrode photo"
                messages.append(f"This doesn't look like an electrode photo ({reason}) -- retake a photo of the SPE strip itself.")
            if vision["angled"]:
                issues.append("angled")
                reason = vision.get("angle_reason") or "the electrode is tilted relative to the frame"
                messages.append(f"Electrode appears angled ({reason}) -- hold the phone/camera directly above the strip, parallel to the surface.")
            if vision["overlapping"]:
                issues.append("overlapping_electrodes")
                reason = vision.get("overlap_reason") or "electrodes appear to be touching or crossing"
                messages.append(f"Electrodes appear to overlap ({reason}) -- lay strips flat and separated before photographing.")
            if vision["multiple_electrodes"]:
                issues.append("multiple_electrodes")
                reason = vision.get("multiple_electrodes_reason") or "more than one electrode is visible in frame"
                messages.append(f"Multiple electrodes are visible ({reason}) but this analysis expects one -- photograph a single strip, or specify which one.")
            if vision["flipped"]:
                issues.append("flipped")
                reason = vision.get("flipped_reason") or "the strip appears to be shown from its reverse side"
                messages.append(f"Electrode appears flipped/reversed ({reason}) -- photograph the printed front side, showing the electrode pads.")
            if vision["tampered"]:
                issues.append("tampered")
                reason = vision.get("tampered_reason") or "visible signs of physical damage or alteration"
                messages.append(f"Electrode shows possible physical tampering/damage ({reason}) -- confirm the strip's integrity before analyzing, or use an undamaged one.")
            if vision["mixed_types"]:
                issues.append("mixed_electrode_types")
                reason = vision.get("mixed_types_reason") or "electrodes of different types/designs appear mixed in frame"
                messages.append(f"Electrodes of different types appear mixed in this photo ({reason}) -- photograph electrodes of the same type together, or one at a time.")

    framing_ok = len(issues) == 0
    return {
        "framing_ok": framing_ok,
        "issues": issues,
        "user_message": " ".join(messages) if messages else "Framing looks good.",
        "proceed": framing_ok,
        "measurements": {
            "resolution": resolution, "blur": blur, "lighting": lighting, "off_centre": off_centre,
            "vision": vision,
        },
    }
