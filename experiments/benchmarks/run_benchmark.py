#!/usr/bin/env python3
"""
experiments/benchmarks/run_benchmark.py

End-to-end benchmark runner for VedicTHG-style pipeline using:
  - texts.json (clip.wav -> transcript)
  - viseme mapping (text -> phonemes -> viseme sequence)
  - renderer (face + mouth sprites -> mp4)
  - profiling (wall time, CPU%, memory delta, effective FPS)

Outputs:
  <out_dir>/per_clip.csv
  <out_dir>/summary.csv
  <out_dir>/table.tex

Typical usage (after creating dummy assets):
  python scripts/prepare_dummy_assets.py
  python experiments/benchmarks/run_benchmark.py \
    --face data/raw/avatar.png \
    --mouth-dir data/raw/visemes \
    --audio-dir data/raw/benchmark \
    --texts-json data/raw/benchmark/texts.json \
    --out-dir results/benchmarks/demo \
    --fps 30 --overlap 0.5 --use-vedic

Notes:
- If `pronouncing` is installed, we use CMUdict word->ARPABET.
- If not installed, we fall back to a simple heuristic phoneme mapper.


python experiments/benchmarks/run_benchmark.py \
  --config configs/benchmark.yaml \
  --out results/bench_YYYYMMDD

python experiments\benchmarks\run_benchmark.py `
  --face data\raw\my_face.jpg `
  --mouth-dir data\raw\visemes `
  --audio-dir data\raw `
  --texts-json data\raw\my_texts.json `
  --out-dir results\myface_dynamic `
  --fps 30 --overlap 0.5 --use-vedic

"""

import argparse
import csv
import json
import math
import os
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Optional (better profiling)
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# Project imports
from vedicthg.viseme_mapper import phoneme_to_viseme_sequence
from vedicthg.render_engine import render_viseme_sequence


# -----------------------------
# Audio helpers
# -----------------------------
def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        return float(n_frames) / float(sr)


# -----------------------------
# Text -> phonemes (ARPABET)
# -----------------------------
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _try_import_pronouncing():
    try:
        import pronouncing  # type: ignore
        return pronouncing
    except Exception:
        return None


_PRON = _try_import_pronouncing()


def _strip_stress(arpabet: str) -> str:
    # AH0 -> AH
    return re.sub(r"\d$", "", arpabet)


def text_to_phonemes(text: str) -> List[str]:
    """
    Convert text to a rough ARPABET phoneme list.

    Best: uses CMUdict via `pronouncing`.
    Fallback: a small heuristic mapping (not linguistically perfect, but deterministic).
    """
    words = _WORD_RE.findall(text.lower())
    phonemes: List[str] = []

    if _PRON is not None:
        for w in words:
            phones_list = _PRON.phones_for_word(w)
            if phones_list:
                # take first pronunciation
                p = phones_list[0].split()
                phonemes.extend([_strip_stress(x) for x in p])
            else:
                phonemes.extend(_heuristic_word_to_phones(w))
        return phonemes

    # fallback only
    for w in words:
        phonemes.extend(_heuristic_word_to_phones(w))
    return phonemes


# Very small heuristic fallback
_VOWELS = {
    "a": "AH",
    "e": "EH",
    "i": "IH",
    "o": "OW",
    "u": "UH",
    "y": "IY",
}
_CONS = {
    "b": "B", "c": "K", "d": "D", "f": "F", "g": "G", "h": "HH",
    "j": "JH", "k": "K", "l": "L", "m": "M", "n": "N", "p": "P",
    "q": "K", "r": "R", "s": "S", "t": "T", "v": "V", "w": "W",
    "x": "K", "z": "Z",
}


def _heuristic_word_to_phones(w: str) -> List[str]:
    out: List[str] = []
    w = w.strip("'")
    if not w:
        return out

    # handle common digraphs quickly
    w = w.replace("th", " TH ")
    w = w.replace("sh", " SH ")
    w = w.replace("ch", " CH ")
    w = w.replace("ng", " NG ")
    w = w.replace("ph", " F ")

    tokens = w.split()
    if len(tokens) > 1:
        for t in tokens:
            out.extend(_heuristic_word_to_phones(t))
        return out

    for ch in w:
        if ch in _VOWELS:
            out.append(_VOWELS[ch])
        elif ch in _CONS:
            out.append(_CONS[ch])

    # ensure at least one phoneme
    if not out:
        out = ["AH"]
    return out


# -----------------------------
# Timing: phonemes -> (start,end)
# -----------------------------
def uniform_phoneme_times(n: int, duration_s: float) -> List[Tuple[float, float]]:
    if n <= 0:
        return []
    dt = duration_s / float(n)
    times = []
    t0 = 0.0
    for i in range(n):
        t1 = duration_s if i == n - 1 else (t0 + dt)
        times.append((t0, t1))
        t0 = t1
    return times


# -----------------------------
# Profiling
# -----------------------------
@dataclass
class Profile:
    wall_s: float
    cpu_pct: float
    mem_delta_mb: float


def profile_call(fn, *args, **kwargs) -> Tuple[object, Profile]:
    """
    Profile a single function call.
    Uses psutil if available; otherwise returns wall-time only.
    """
    if psutil is None:
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        wall = time.perf_counter() - t0
        return out, Profile(wall_s=wall, cpu_pct=float("nan"), mem_delta_mb=float("nan"))

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    cpu0 = proc.cpu_times().user + proc.cpu_times().system
    t0 = time.perf_counter()

    out = fn(*args, **kwargs)

    wall = time.perf_counter() - t0
    cpu1 = proc.cpu_times().user + proc.cpu_times().system
    rss1 = proc.memory_info().rss

    cpu_pct = ((cpu1 - cpu0) / wall) * 100.0 if wall > 0 else 0.0
    mem_mb = (rss1 - rss0) / (1024 * 1024)
    return out, Profile(wall_s=wall, cpu_pct=cpu_pct, mem_delta_mb=mem_mb)


# -----------------------------
# Benchmark core
# -----------------------------
def safe_slug(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return name[:180] if len(name) > 180 else name


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary(path: Path, rows: List[Dict[str, object]]):
    """
    Writes simple mean summary for numeric columns.
    """
    if not rows:
        return

    numeric_keys = []
    for k, v in rows[0].items():
        if isinstance(v, (int, float)) and k not in ("clip_index",):
            numeric_keys.append(k)

    def _mean(key: str) -> float:
        vals = []
        for r in rows:
            v = r.get(key)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                vals.append(float(v))
        return float(sum(vals) / len(vals)) if vals else float("nan")

    summary = {"n_clips": len(rows)}
    for k in numeric_keys:
        summary[f"mean_{k}"] = _mean(k)

    write_csv(path, [summary], fieldnames=list(summary.keys()))


def write_latex_table(path: Path, summary_csv: Path):
    """
    Produces a small LaTeX table with key metrics.
    """
    # read summary.csv
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = next(reader)

    def g(key: str) -> str:
        v = data.get(key, "")
        return v if v != "" else "N/A"

    tex = r"""\begin{table}[t]
\centering
\caption{Benchmark summary (mean across clips).}
\label{tab:benchmark_summary}
\begin{tabular}{l c}
\toprule
Metric & Value \\
\midrule
Number of clips & %s \\
Mean audio duration (s) & %s \\
Mean render wall time (s) & %s \\
Mean effective FPS (frames/s) & %s \\
Mean ms/frame & %s \\
Mean CPU usage (\%%) & %s \\
Mean memory delta (MB) & %s \\
\bottomrule
\end{tabular}
\end{table}
""" % (
        g("n_clips"),
        g("mean_audio_s"),
        g("mean_wall_s"),
        g("mean_eff_fps"),
        g("mean_ms_per_frame"),
        g("mean_cpu_pct"),
        g("mean_mem_delta_mb"),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", required=True, type=str, help="Base face image (png)")
    ap.add_argument("--mouth-dir", required=True, type=str, help="Mouth sprites dir containing 0.png..14.png")
    ap.add_argument("--audio-dir", required=True, type=str, help="Directory with wav clips")
    ap.add_argument("--texts-json", required=True, type=str, help="JSON mapping: wav_filename -> transcript")
    ap.add_argument("--out-dir", required=True, type=str, help="Output directory for results")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--use-vedic", action="store_true", help="Enable vedic modulation in viseme blending")
    ap.add_argument("--vedic-strength", type=float, default=0.15)
    ap.add_argument("--mouth-x", type=int, default=100)
    ap.add_argument("--mouth-y", type=int, default=300)
    ap.add_argument("--resolution", type=str, default="", help='Optional "WxH" (e.g., 512x512)')
    ap.add_argument("--max-clips", type=int, default=0, help="If >0, limit number of clips")
    ap.add_argument("--no-video", action="store_true", help="Skip video writing (profiles mapping only)")
    args = ap.parse_args()

    face_path = Path(args.face)
    mouth_dir = Path(args.mouth_dir)
    audio_dir = Path(args.audio_dir)
    texts_json = Path(args.texts_json)
    out_dir = Path(args.out_dir)
    out_videos = out_dir / "videos"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_videos.mkdir(parents=True, exist_ok=True)

    # parse resolution
    resolution: Optional[Tuple[int, int]] = None
    if args.resolution:
        m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", args.resolution.lower())
        if not m:
            raise ValueError('Invalid --resolution. Use "WxH" like 512x512')
        resolution = (int(m.group(1)), int(m.group(2)))

    # load texts map
    with open(texts_json, "r", encoding="utf-8") as f:
        mapping: Dict[str, str] = json.load(f)

    items = sorted(mapping.items(), key=lambda kv: kv[0])
    if args.max_clips and args.max_clips > 0:
        items = items[: args.max_clips]

    rows: List[Dict[str, object]] = []

    for idx, (wav_name, text) in enumerate(items, start=1):
        wav_path = audio_dir / wav_name
        if not wav_path.exists():
            print(f"[skip] missing wav: {wav_path}")
            continue

        audio_s = wav_duration_seconds(wav_path)
        phonemes = text_to_phonemes(text)
        times = uniform_phoneme_times(len(phonemes), audio_s)

        vis_seq = phoneme_to_viseme_sequence(
            phonemes=phonemes,
            times=times,
            overlap=args.overlap,
            use_vedic=args.use_vedic,
            vedic_strength=args.vedic_strength,
        )

        total_frames = int(math.ceil(audio_s * args.fps))
        out_video = out_videos / f"{safe_slug(Path(wav_name).stem)}.mp4"

        # profile either full render or only mapping
        if args.no_video:
            _, prof = profile_call(lambda: None)
        else:
            _, prof = profile_call(
                render_viseme_sequence,
                viseme_seq=vis_seq,
                face_image_path=str(face_path),
                mouth_sprites_dir=str(mouth_dir),
                out_video_path=str(out_video),
                fps=args.fps,
                # mouth_position=(args.mouth_x, args.mouth_y),
                # resolution=resolution,
                audio_path=str(wav_path),  # mux audio for realism (optional)
            )

        eff_fps = (total_frames / prof.wall_s) if prof.wall_s > 0 else 0.0
        ms_per_frame = (1000.0 * prof.wall_s / total_frames) if total_frames > 0 else 0.0

        row = {
            "clip_index": idx,
            "wav": wav_name,
            "text": text,
            "audio_s": round(audio_s, 6),
            "n_phonemes": len(phonemes),
            "n_viseme_segments": len(vis_seq),
            "fps": args.fps,
            "overlap": args.overlap,
            "use_vedic": int(bool(args.use_vedic)),
            "vedic_strength": args.vedic_strength if args.use_vedic else 0.0,
            "wall_s": round(prof.wall_s, 6),
            "cpu_pct": round(prof.cpu_pct, 6) if isinstance(prof.cpu_pct, float) else prof.cpu_pct,
            "mem_delta_mb": round(prof.mem_delta_mb, 6) if isinstance(prof.mem_delta_mb, float) else prof.mem_delta_mb,
            "total_frames": total_frames,
            "eff_fps": round(eff_fps, 6),
            "ms_per_frame": round(ms_per_frame, 6),
            "out_video": str(out_video) if not args.no_video else "",
        }
        rows.append(row)
        print(f"[{idx}/{len(items)}] {wav_name}: wall={row['wall_s']}s  eff_fps={row['eff_fps']}  cpu={row['cpu_pct']}")

    # write outputs
    per_clip_csv = out_dir / "per_clip.csv"
    summary_csv = out_dir / "summary.csv"
    table_tex = out_dir / "table.tex"

    if rows:
        fieldnames = list(rows[0].keys())
        write_csv(per_clip_csv, rows, fieldnames=fieldnames)
        write_summary(summary_csv, rows)
        write_latex_table(table_tex, summary_csv)
        print("\nWritten:")
        print(f"  - {per_clip_csv}")
        print(f"  - {summary_csv}")
        print(f"  - {table_tex}")
    else:
        print("[error] No rows produced (check paths and texts.json).")


if __name__ == "__main__":
    main()

