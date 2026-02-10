import os
import json
import time
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
'''python experiments\ablation\generate_ablation_tables.py `
  --face data\raw\my_face.jpg `
  --audio data\raw\benchmark\clip_01.wav `
  --mouth-bank data\raw\mouth_bank `
  --phonemes-json data\raw\benchmark\phonemes_clip01.json `
  --out-dir results\ablations `
  --fps 30
'''
# --- Project imports ---
from vedicthg.landmarks import detect_facemesh_landmarks, build_dynamic_rig
from vedicthg.audio_motion import build_motion_signals
from vedicthg.warps import polygon_mask, apply_local_shift, apply_masked_global_motion, apply_global_affine
from vedicthg.viseme_mapper import phoneme_to_viseme_sequence





# -----------------------------
# Constants (same as your render)
# -----------------------------
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]


# -----------------------------
# Optional identity metric (InsightFace)
# -----------------------------
def try_load_face_embedder():
    """
    Optional: identity drift via InsightFace (ArcFace).
    If not installed, return None.
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        return app
    except Exception:
        return None


def identity_embedding(app, bgr: np.ndarray) -> Optional[np.ndarray]:
    if app is None:
        return None
    faces = app.get(bgr)
    if not faces:
        return None
    emb = faces[0].normed_embedding.astype(np.float32)
    return emb


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_mouth_bank(mouth_bank_dir: str, n: int = 15) -> Dict[int, np.ndarray]:
    bank = {}
    for i in range(n):
        p = os.path.join(mouth_bank_dir, f"{i}.png")
        if os.path.exists(p):
            rgba = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2] != 4:
                raise RuntimeError(f"Bad mouth bank PNG (needs RGBA): {p}")
            bank[i] = rgba
    if not bank:
        raise FileNotFoundError(f"No mouth bank PNGs found in: {mouth_bank_dir}")

    first = bank[min(bank.keys())]
    for i in range(n):
        bank.setdefault(i, first)
    return bank


# -----------------------------
# Blending (fixed broadcasting bug)
# -----------------------------
def alpha_blend_rgba_over_bgr(
    dst_bgr: np.ndarray,
    src_rgba: np.ndarray,               # HxWx4 uint8
    bbox: Tuple[int, int, int, int],
    mask_full: np.ndarray,              # HxW float 0..1
    alpha_gain: float = 0.65,
    blur_sigma: float = 2.0,
    ellipse_strength: float = 1.0,
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(dst_bgr.shape[1], x1); y1 = min(dst_bgr.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return dst_bgr

    roi = dst_bgr[y0:y1, x0:x1].astype(np.float32)

    src = src_rgba.astype(np.float32)
    src_bgr = src[..., :3][..., ::-1]            # RGBA->BGR
    a = (src[..., 3] / 255.0).astype(np.float32) # HxW

    m = mask_full[y0:y1, x0:x1].astype(np.float32)  # HxW

    # ellipse feather to remove bbox edges
    hh, ww = a.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    cx, cy = ww / 2.0, hh / 2.0
    rx, ry = max(2.0, 0.48 * ww), max(2.0, 0.48 * hh)
    e = 1.0 - np.clip(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2, 0.0, 1.0)
    e = np.clip(e, 0.0, 1.0)
    if ellipse_strength < 1.0:
        e = (1.0 - ellipse_strength) + ellipse_strength * e

    a = a * m * e
    a *= float(alpha_gain)
    a = np.clip(a, 0.0, 1.0)

    if blur_sigma and blur_sigma > 0:
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    a3 = a[..., None]  # HxWx1
    out = src_bgr * a3 + roi * (1.0 - a3)

    dst = dst_bgr.copy()
    dst[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    return dst


# -----------------------------
# Minimal renderer (variants supported)
# -----------------------------
@dataclass
class RenderVariant:
    name: str
    enable_rig: bool = True
    enable_jaw: bool = True
    enable_cheeks: bool = True
    smooth_bbox: bool = True
    move_head_only: bool = True
    use_global_affine: bool = False  # â† ADDED vineet
    head_gain: float = 0.18

    # mouth blend params
    blend_mask_type: str = "inner"   # "rect" | "outer" | "inner"
    feather_px: int = 25
    ellipse_strength: float = 1.0
    blur_sigma: float = 2.0
    alpha_gain: float = 0.65


def build_masks_from_landmarks(face_bgr: np.ndarray):
    lm = detect_facemesh_landmarks(face_bgr)
    rig = build_dynamic_rig(lm)
    h, w = face_bgr.shape[:2]

    jaw_poly = np.array([lm.get(61), lm.get(291), lm.get(152)], dtype=np.float32)
    cheek_l = np.array([lm.get(234), lm.get(61), lm.get(13)], dtype=np.float32)
    cheek_r = np.array([lm.get(291), lm.get(454), lm.get(13)], dtype=np.float32)

    rig["jaw_mask"] = polygon_mask((h, w), jaw_poly, feather=35).astype(np.float32)
    rig["cheek_l"] = polygon_mask((h, w), cheek_l, feather=35).astype(np.float32)
    rig["cheek_r"] = polygon_mask((h, w), cheek_r, feather=35).astype(np.float32)

    inner_poly = np.array([lm.get(i) for i in LIPS_INNER], dtype=np.float32)
    rig["inner_mouth_mask"] = polygon_mask((h, w), inner_poly, feather=25).astype(np.float32)

    face_poly = np.array([lm.get(i) for i in FACE_OVAL], dtype=np.float32)
    rig["head_mask"] = polygon_mask((h, w), face_poly, feather=70).astype(np.float32)

    return rig, lm


def render_video_variant(
    variant: RenderVariant,
    face_path: str,
    audio_path: str,
    viseme_seq: List[Tuple[int, float, float, np.ndarray]],
    mouth_bank_dir: str,
    out_mp4: str,
    fps: int = 30,
) -> Dict[str, Any]:
    ensure_dir(os.path.dirname(out_mp4))

    face_bgr = cv2.imread(face_path, cv2.IMREAD_COLOR)
    if face_bgr is None:
        raise FileNotFoundError(f"Cannot read face image: {face_path}")

    mouth_bank = load_mouth_bank(mouth_bank_dir, n=15)

    rig = None
    if variant.enable_rig:
        rig, lm = build_masks_from_landmarks(face_bgr)

    motions = build_motion_signals(audio_path, viseme_seq, fps) if audio_path else None

    total_t = max(e for _, _, e, _ in viseme_seq)
    ends = [e for _, _, e, _ in viseme_seq]
    vis_ids = [v for v, _, _, _ in viseme_seq]
    seg_idx = 0
    prev_bbox = None

    # choose mask for blending
    def select_blend_mask():
        if rig is None:
            return None
        if variant.blend_mask_type == "inner":
            return rig["inner_mouth_mask"]
        # "outer" not implemented here -> fallback to inner
        if variant.blend_mask_type == "outer":
            return rig["inner_mouth_mask"]
        # rect -> use all-ones ROI mask later
        return rig["inner_mouth_mask"]

    blend_mask_full = select_blend_mask()

    def update_segment_index(t):
        nonlocal seg_idx
        while seg_idx + 1 < len(ends) and ends[seg_idx] <= t:
            seg_idx += 1

    def default_mouth_bbox(h, w):
        # fallback bbox if rig disabled
        cx, cy = int(0.5 * w), int(0.62 * h)
        bw, bh = int(0.38 * w), int(0.18 * h)
        return (cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2)

    # render loop (write mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = face_bgr.shape[:2]
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))

    n_frames = int(math.ceil(total_t * fps))
    t0 = time.time()
    for fi in range(n_frames):
        t = fi / fps
        update_segment_index(t)
        vis = int(vis_ids[seg_idx])
        frame = face_bgr.copy()

        if rig is not None and motions is not None:
            eng = float(np.clip(motions.energy[min(fi, len(motions.energy) - 1)], 0.0, 0.6))

            # head-only motion (background fixed)
            # if variant.move_head_only and variant.head_gain > 0:
            #     frame = apply_masked_global_motion(
            #         frame,
            #         rig["head_mask"],
            #         dx=variant.head_gain * float(rig["head_px"]) * math.sin(0.25 * t) * eng,
            #         dy=0.0,
            #         rot_deg=variant.head_gain * float(rig["head_rot_deg"]) * math.sin(0.25 * t) * eng,
            #     )
            # replaced by below:
            dx = variant.head_gain * float(rig["head_px"]) * math.sin(0.25 * t) * eng
            rot = variant.head_gain * float(rig["head_rot_deg"]) * math.sin(0.25 * t) * eng

            if variant.use_global_affine:
                # H1: full-frame motion (background moves)
                frame = apply_global_affine(frame, dx=dx, dy=0.0, rot_deg=rot)

            elif variant.move_head_only:
                # H2: head-only motion (background stable)
                frame = apply_masked_global_motion(
                    frame,
                    rig["head_mask"],
                    dx=dx,
                    dy=0.0,
                    rot_deg=rot,
                )

            # jaw
            if variant.enable_jaw:
                jaw_open = float(motions.jaw_open[min(fi, len(motions.jaw_open) - 1)])
                frame = apply_local_shift(
                    frame,
                    rig["jaw_mask"],
                    (0.0, float(rig["jaw_px"]) * jaw_open),
                )

            # cheeks
            if variant.enable_cheeks:
                sm = float(motions.smile[min(fi, len(motions.smile) - 1)])
                cheek = float(rig["cheek_px"]) * (0.6 * sm + 0.4 * eng)
                frame = apply_local_shift(frame, rig["cheek_l"], (0.0, -cheek))
                frame = apply_local_shift(frame, rig["cheek_r"], (0.0, -cheek))

        # mouth compositing (works with/without rig)
        if rig is not None:
            bbox = np.array(rig["mouth_bbox"], dtype=np.float32)
            if variant.smooth_bbox:
                nonlocal_prev = prev_bbox
                if nonlocal_prev is None:
                    nonlocal_prev = bbox.copy()
                nonlocal_prev = 0.85 * nonlocal_prev + 0.15 * bbox
                prev_bbox = nonlocal_prev
                x0, y0, x1, y1 = nonlocal_prev.astype(int).tolist()
            else:
                x0, y0, x1, y1 = bbox.astype(int).tolist()
        else:
            x0, y0, x1, y1 = default_mouth_bbox(h, w)

        patch = mouth_bank.get(vis, mouth_bank[0])
        pw, ph = max(2, x1 - x0), max(2, y1 - y0)
        patch_rs = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)

        if blend_mask_full is None:
            mask_full = np.ones((h, w), dtype=np.float32)
        else:
            mask_full = blend_mask_full

        frame = alpha_blend_rgba_over_bgr(
            dst_bgr=frame,
            src_rgba=patch_rs,
            bbox=(x0, y0, x1, y1),
            mask_full=mask_full,
            alpha_gain=variant.alpha_gain,
            blur_sigma=variant.blur_sigma,
            ellipse_strength=variant.ellipse_strength,
        )

        writer.write(frame)

    writer.release()
    render_time = time.time() - t0

    return {
        "frames": n_frames,
        "duration_s": total_t,
        "render_time_s": render_time,
        "fps_effective": float(n_frames / max(1e-6, render_time)),
        "out_mp4": out_mp4,
    }


# -----------------------------
# Metric extraction from mp4
# -----------------------------
def read_video_frames(path: str, max_frames: Optional[int] = None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        i += 1
        if max_frames and i >= max_frames:
            break
    cap.release()
    return frames


def compute_mouth_roi_bbox_from_rig(rig: Dict[str, Any]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = rig["mouth_bbox"]
    return int(x0), int(y0), int(x1), int(y1)


def compute_flicker(frames: List[np.ndarray], bbox: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    diffs = []
    prev = None
    for fr in frames:
        roi = fr[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(np.mean(np.abs(g - prev)))
        prev = g
    if not diffs:
        return float("nan")
    return float(np.var(diffs))


def compute_seam_score(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return float("nan")
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    # boundary band only
    b = 2
    top = mag[:b, :].mean()
    bot = mag[-b:, :].mean()
    left = mag[:, :b].mean()
    right = mag[:, -b:].mean()
    return float((top + bot + left + right) / 4.0)


def compute_teeth_detail(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return float("nan")
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # high-frequency energy via Laplacian variance
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float(np.var(lap))


def compute_background_stability(frames: List[np.ndarray], head_mask: np.ndarray) -> float:
    """
    Simple stability score: mean abs difference in background (1 - head_mask).
    lower diff => higher stability; we return "stability" as (1 / (1+diff)).
    """
    if head_mask is None:
        return float("nan")
    bg = (1.0 - head_mask).astype(np.float32)
    bg3 = bg[..., None]
    prev = None
    diffs = []
    for fr in frames:
        f = fr.astype(np.float32)
        bgpix = f * bg3
        if prev is not None:
            diffs.append(np.mean(np.abs(bgpix - prev)))
        prev = bgpix
    if not diffs:
        return float("nan")
    d = float(np.mean(diffs))
    return float(1.0 / (1.0 + d))


def compute_sync_error_simple(viseme_seq, fps, motions) -> float:
    """
    Lightweight sync metric:
    For each viseme segment, find the peak jaw_open inside it.
    Sync error is |peak_time - segment_midpoint| averaged (ms).
    """
    if motions is None:
        return float("nan")

    jaw = motions.jaw_open
    total = 0.0
    cnt = 0

    for (v, t0, t1, _) in viseme_seq:
        if t1 <= t0:
            continue
        i0 = int(max(0, math.floor(t0 * fps)))
        i1 = int(min(len(jaw) - 1, math.ceil(t1 * fps)))
        if i1 <= i0:
            continue
        seg = jaw[i0:i1]
        if len(seg) < 2:
            continue
        pk = int(np.argmax(seg))
        peak_t = (i0 + pk) / fps
        mid_t = 0.5 * (t0 + t1)
        total += abs(peak_t - mid_t)
        cnt += 1

    if cnt == 0:
        return float("nan")
    return float((total / cnt) * 1000.0)


# -----------------------------
# Variant sets for the 4 tables
# -----------------------------
def table_A1_variants():
    return [
        RenderVariant(name="A0_Base_NoRig", enable_rig=False, enable_jaw=False, enable_cheeks=False, smooth_bbox=False, move_head_only=False),
        RenderVariant(name="A1_+Rig_Jaw", enable_rig=True, enable_jaw=True, enable_cheeks=False, smooth_bbox=False, move_head_only=False),
        RenderVariant(name="A2_+Cheeks", enable_rig=True, enable_jaw=True, enable_cheeks=True, smooth_bbox=False, move_head_only=False),
        RenderVariant(name="A3_+BBoxEMA", enable_rig=True, enable_jaw=True, enable_cheeks=True, smooth_bbox=True, move_head_only=False),
        RenderVariant(name="A4_+HeadOnly", enable_rig=True, enable_jaw=True, enable_cheeks=True, smooth_bbox=True, move_head_only=True, head_gain=0.18),
    ]


def table_A2_variants(overlaps=(0.0, 0.3, 0.5, 0.5, 0.7)):
    # Here "Vedic Blend" is represented via overlap choice and using your mapper (you can tag rows)
    # We output overlap as a column; rendering variant stays same
    base = RenderVariant(name="OverlapTest", enable_rig=True, enable_jaw=True, enable_cheeks=True, smooth_bbox=True, move_head_only=True)
    return [(base, overlaps[0], False),
            (base, overlaps[1], False),
            (base, overlaps[2], False),
            (base, overlaps[3], True),
            (base, overlaps[4], True)]


def table_A3_variants():
    # B0..B3
    return [
        RenderVariant(name="B0_Rect_NoFeather", enable_rig=True, blend_mask_type="inner", feather_px=0, ellipse_strength=0.0, blur_sigma=0.0, alpha_gain=1.0),
        RenderVariant(name="B1_OuterLip", enable_rig=True, blend_mask_type="outer", feather_px=25, ellipse_strength=0.0, blur_sigma=1.0, alpha_gain=0.80),
        RenderVariant(name="B2_Inner_Default", enable_rig=True, blend_mask_type="inner", feather_px=25, ellipse_strength=1.0, blur_sigma=2.0, alpha_gain=0.65),
        RenderVariant(name="B3_Inner_Softer", enable_rig=True, blend_mask_type="inner", feather_px=35, ellipse_strength=1.0, blur_sigma=3.0, alpha_gain=0.55),
    ]


# def table_A4_variants():
#     return [
#         RenderVariant(name="H0_None", enable_rig=True, move_head_only=False, head_gain=0.0),
#         RenderVariant(name="H1_GlobalAffine_NotUsedHere", enable_rig=True, move_head_only=False, head_gain=0.0),  # placeholder (see note below)
#         RenderVariant(name="H2_HeadOnlyMasked", enable_rig=True, move_head_only=True, head_gain=0.18),
#     ]
def table_A4_variants():
    return [
        RenderVariant(
            name="H0_None",
            enable_rig=True,
            move_head_only=False,
            use_global_affine=False,
            head_gain=0.0
        ),
        RenderVariant(
            name="H1_GlobalAffine",
            enable_rig=True,
            move_head_only=False,
            use_global_affine=True,   # â† TRUE
            head_gain=0.18
        ),
        RenderVariant(
            name="H2_HeadOnlyMasked",
            enable_rig=True,
            move_head_only=True,
            use_global_affine=False,
            head_gain=0.18
        ),
    ]


# -----------------------------
# Main routine
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", required=True, help="Path to face image (BGR-readable: png/jpg).")
    ap.add_argument("--audio", required=True, help="Path to wav file.")
    ap.add_argument("--mouth-bank", required=True, help="Dir containing 0.png..14.png (RGBA).")
    ap.add_argument("--out-dir", default="results/ablations", help="Output directory.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-frames-metrics", type=int, default=300, help="Cap frames read for metric computation.")
    ap.add_argument("--phonemes-json", default=None,
                    help="Optional JSON with {'phonemes':[...], 'times':[[t0,t1],...]}. If not provided, uses a tiny dummy example.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # -------- Build viseme sequence --------
    if args.phonemes_json and os.path.exists(args.phonemes_json):
        with open(args.phonemes_json, "r", encoding="utf-8") as f:
            j = json.load(f)
        phonemes = j["phonemes"]
        times = [tuple(x) for x in j["times"]]
    else:
        # Fallback dummy (you should provide phonemes_json for real evaluation)
        phonemes = ["P", "AH", "T", "M"]
        times = [(0.0, 0.25), (0.25, 0.55), (0.55, 0.75), (0.75, 1.10)]

    # We will re-create viseme_seq per overlap for table A2
    viseme_seq_default = phoneme_to_viseme_sequence(phonemes, times, overlap=0.5)

    # Prepare optional identity embedder
    face_app = try_load_face_embedder()
    face_bgr = cv2.imread(args.face, cv2.IMREAD_COLOR)
    ref_emb = identity_embedding(face_app, face_bgr) if face_bgr is not None else None

    # For metric bbox/masks, build rig once (if face detected)
    rig = None
    try:
        rig, _ = build_masks_from_landmarks(face_bgr)
    except Exception:
        rig = None

    # Helper to compute metrics for a rendered mp4
    def compute_metrics(mp4_path: str, fps_effective: float, motions_obj, viseme_seq_used):
        frames = read_video_frames(mp4_path, max_frames=args.max_frames_metrics)
        if not frames:
            raise RuntimeError(f"No frames read from {mp4_path}")

        if rig is not None:
            bbox = compute_mouth_roi_bbox_from_rig(rig)
            head_mask = rig.get("head_mask", None)
        else:
            h, w = frames[0].shape[:2]
            bbox = (int(0.31*w), int(0.53*h), int(0.69*w), int(0.71*h))
            head_mask = None

        flicker = compute_flicker(frames, bbox)
        seam = compute_seam_score(frames[len(frames)//2], bbox)
        teeth = compute_teeth_detail(frames[len(frames)//2], bbox)
        bg_stab = compute_background_stability(frames, head_mask) if head_mask is not None else float("nan")
        sync_err = compute_sync_error_simple(viseme_seq_used, args.fps, motions_obj)

        # identity drift (optional): average over a few frames
        id_drift = float("nan")
        if face_app is not None and ref_emb is not None:
            picks = [0, len(frames)//2, len(frames)-1]
            ds = []
            for pi in picks:
                emb = identity_embedding(face_app, frames[pi])
                if emb is not None:
                    ds.append(cosine_distance(ref_emb, emb))
            if ds:
                id_drift = float(np.mean(ds))

        return {
            "fps_effective": fps_effective,
            "sync_err_ms": sync_err,
            "flicker_var": flicker,
            "seam_score": seam,
            "teeth_detail": teeth,
            "bg_stability": bg_stab,
            "id_drift": id_drift,
        }

    # -------------------------
    # TABLE A1
    # -------------------------
    rows_A1 = []
    for v in table_A1_variants():
        out_mp4 = os.path.join(args.out_dir, f"{v.name}.mp4")
        motions = build_motion_signals(args.audio, viseme_seq_default, args.fps)

        prof = render_video_variant(
            variant=v,
            face_path=args.face,
            audio_path=args.audio,
            viseme_seq=viseme_seq_default,
            mouth_bank_dir=args.mouth_bank,
            out_mp4=out_mp4,
            fps=args.fps,
        )
        m = compute_metrics(out_mp4, prof["fps_effective"], motions, viseme_seq_default)
        rows_A1.append({
            "variant": v.name,
            "mouth_bank": True,
            "bbox_smooth": v.smooth_bbox,
            "jaw_warp": v.enable_jaw,
            "cheek_warp": v.enable_cheeks,
            "head_only": v.move_head_only and v.head_gain > 0,
            **m
        })

    save_A1 = os.path.join(args.out_dir, "ablation_A1_components.csv")
    pd_dump(save_A1, rows_A1)

    # -------------------------
    # TABLE A2 (overlap)
    # -------------------------
    rows_A2 = []
    for (v, ov, vedic_flag) in table_A2_variants():
        viseme_seq_ov = phoneme_to_viseme_sequence(phonemes, times, overlap=float(ov))
        motions = build_motion_signals(args.audio, viseme_seq_ov, args.fps)

        out_mp4 = os.path.join(args.out_dir, f"A2_overlap_{ov}_{'vedic' if vedic_flag else 'plain'}.mp4")
        prof = render_video_variant(v, args.face, args.audio, viseme_seq_ov, args.mouth_bank, out_mp4, fps=args.fps)
        m = compute_metrics(out_mp4, prof["fps_effective"], motions, viseme_seq_ov)

        rows_A2.append({
            "overlap": float(ov),
            "vedic_blend": bool(vedic_flag),
            "sync_err_ms": m["sync_err_ms"],
            "flicker_var": m["flicker_var"],
            "fps_effective": m["fps_effective"],
        })

    save_A2 = os.path.join(args.out_dir, "ablation_A2_overlap.csv")
    pd_dump(save_A2, rows_A2)

    # -------------------------
    # TABLE A3 (mouth blending)
    # -------------------------
    rows_A3 = []
    for v in table_A3_variants():
        out_mp4 = os.path.join(args.out_dir, f"{v.name}.mp4")
        motions = build_motion_signals(args.audio, viseme_seq_default, args.fps)

        prof = render_video_variant(v, args.face, args.audio, viseme_seq_default, args.mouth_bank, out_mp4, fps=args.fps)
        m = compute_metrics(out_mp4, prof["fps_effective"], motions, viseme_seq_default)

        rows_A3.append({
            "variant": v.name,
            "blend_mask": v.blend_mask_type,
            "feather_px": v.feather_px,
            "ellipse_strength": v.ellipse_strength,
            "blur_sigma": v.blur_sigma,
            "alpha_gain": v.alpha_gain,
            "seam_score": m["seam_score"],
            "teeth_detail": m["teeth_detail"],
        })

    save_A3 = os.path.join(args.out_dir, "ablation_A3_mouthblend.csv")
    pd_dump(save_A3, rows_A3)

    # -------------------------
    # TABLE A4 (head motion strategies)
    # NOTE: "Global affine" variant is not re-implemented here because your current pipeline
    # uses masked head motion. If you still want H1 true-global, I can add it in this same file.
    # -------------------------
    rows_A4 = []
    for v in table_A4_variants():
        out_mp4 = os.path.join(args.out_dir, f"{v.name}.mp4")
        motions = build_motion_signals(args.audio, viseme_seq_default, args.fps)

        prof = render_video_variant(v, args.face, args.audio, viseme_seq_default, args.mouth_bank, out_mp4, fps=args.fps)
        m = compute_metrics(out_mp4, prof["fps_effective"], motions, viseme_seq_default)

        # rows_A4.append({
        #     "variant": v.name,
        #     "motion_type": "none" if v.head_gain <= 0 else ("head_only_masked" if v.move_head_only else "global_affine"),
        #     "translate": v.head_gain > 0,
        #     "rotate": v.head_gain > 0,
        #     "bg_stability": m["bg_stability"],
        #     "flicker_var": m["flicker_var"],
        #     "fps_effective": m["fps_effective"],
        # })
        rows_A4.append({
            "variant": v.name,
            "motion_type": (
                "none" if v.head_gain <= 0 else
                "global_affine" if v.use_global_affine else
                "head_only_masked"
            ),
            "translate": v.head_gain > 0,
            "rotate": v.head_gain > 0,
            "bg_stability": m["bg_stability"],
            "flicker_var": m["flicker_var"],
            "fps_effective": m["fps_effective"],
        })

    save_A4 = os.path.join(args.out_dir, "ablation_A4_headmotion.csv")
    pd_dump(save_A4, rows_A4)

    print("\nâœ… Done.")
    print(f"Saved:\n  {save_A1}\n  {save_A2}\n  {save_A3}\n  {save_A4}")
    print("\nTip: Provide --phonemes-json for real sync evaluation (otherwise dummy phonemes are used).")


# -----------------------------
# Minimal CSV writer (no pandas requirement)
# -----------------------------
def pd_dump(csv_path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, bool):
                    v = "1" if v else "0"
                elif v is None:
                    v = ""
                vals.append(str(v))
            f.write(",".join(vals) + "\n")


if __name__ == "__main__":
    main()

