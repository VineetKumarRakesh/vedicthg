#!/usr/bin/env python3
"""
experiments/ablation/run_ablations.py

Runs a small grid of ablations and produces:
  <out_dir>/per_run.csv
  <out_dir>/summary.csv
  <out_dir>/table.tex

It reuses the same core pipeline as run_benchmark.py:
  - texts.json mapping wav->text
  - text->phonemes->viseme sequence
  - render video (optional)
  - profile wall time / CPU% / mem
  - compute effective FPS and ms/frame

Typical usage:
  python scripts/prepare_dummy_assets.py
  python experiments/ablation/run_ablations.py \
    --face data/raw/avatar.png \
    --mouth-dir data/raw/visemes \
    --audio-dir data/raw/benchmark \
    --texts-json data/raw/benchmark/texts.json \
    --out-dir results/ablations/demo \
    --max-clips 10

You can skip video encoding to speed up ablations:
  ... --no-video

Ablations included by default:
  - linear blend (use_vedic=0)
  - vedic blend (use_vedic=1, strength=0.15)
  - overlap sweep: 0.2 / 0.5 / 0.8 (with vedic)
  - fps sweep: 25 / 30 (with vedic)
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
from vedicthg.viseme_mapper import phoneme_to_viseme_sequence
from vedicthg.render_engine import render_viseme_sequence


# -----------------------------
# Utilities shared with benchmark
# -----------------------------
_WORD_RE = re.compile(r"[a-zA-Z']+")


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return float(wf.getnframes()) / float(wf.getframerate())


def _try_import_pronouncing():
    try:
        import pronouncing  # type: ignore
        return pronouncing
    except Exception:
        return None


_PRON = _try_import_pronouncing()


def _strip_stress(arpabet: str) -> str:
    return re.sub(r"\d$", "", arpabet)


_VOWELS = {"a": "AH", "e": "EH", "i": "IH", "o": "OW", "u": "UH", "y": "IY"}
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
    w = w.replace("th", " TH ").replace("sh", " SH ").replace("ch", " CH ").replace("ng", " NG ").replace("ph", " F ")
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
    return out or ["AH"]


def text_to_phonemes(text: str) -> List[str]:
    words = _WORD_RE.findall(text.lower())
    phonemes: List[str] = []
    if _PRON is not None:
        for w in words:
            phones_list = _PRON.phones_for_word(w)
            if phones_list:
                phonemes.extend([_strip_stress(x) for x in phones_list[0].split()])
            else:
                phonemes.extend(_heuristic_word_to_phones(w))
        return phonemes
    for w in words:
        phonemes.extend(_heuristic_word_to_phones(w))
    return phonemes


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


@dataclass
class Profile:
    wall_s: float
    cpu_pct: float
    mem_delta_mb: float


def profile_call(fn, *args, **kwargs) -> Tuple[object, Profile]:
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


def summarize_runs(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Group by run_id and compute mean of numeric columns.
    """
    if not rows:
        return []

    # group
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        rid = str(r["run_id"])
        groups.setdefault(rid, []).append(r)

    summaries: List[Dict[str, object]] = []

    for rid, gr in groups.items():
        # identify numeric keys
        numeric_keys = []
        for k, v in gr[0].items():
            if isinstance(v, (int, float)) and k not in ("clip_index",):
                numeric_keys.append(k)

        def mean(key: str) -> float:
            vals = []
            for rr in gr:
                v = rr.get(key)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    vals.append(float(v))
            return float(sum(vals) / len(vals)) if vals else float("nan")

        s: Dict[str, object] = {
            "run_id": rid,
            "n_clips": len(gr),
            "config_fps": gr[0]["fps"],
            "config_overlap": gr[0]["overlap"],
            "config_use_vedic": gr[0]["use_vedic"],
            "config_vedic_strength": gr[0]["vedic_strength"],
        }
        # metrics
        for k in ["audio_s", "wall_s", "cpu_pct", "mem_delta_mb", "eff_fps", "ms_per_frame"]:
            s[f"mean_{k}"] = mean(k)
        summaries.append(s)

    # sort: baseline first if present, then others
    summaries.sort(key=lambda x: str(x["run_id"]))
    return summaries


def write_latex_table(path: Path, summaries: List[Dict[str, object]]):
    """
    LaTeX table comparing ablations.
    """
    def fmt(v, nd=3):
        if isinstance(v, float):
            if math.isnan(v):
                return "N/A"
            return f"{v:.{nd}f}"
        return str(v)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study across key configuration switches. Values are means across clips.}")
    lines.append(r"\label{tab:ablations}")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Run & FPS & Overlap & Vedic & Eff.\ FPS $\uparrow$ & ms/frame $\downarrow$ \\")
    lines.append(r"\midrule")

    for s in summaries:
        rid = s["run_id"]
        fps = s["config_fps"]
        ov = s["config_overlap"]
        vflag = "Yes" if int(s["config_use_vedic"]) == 1 else "No"
        eff = fmt(s["mean_eff_fps"], 2)
        msf = fmt(s["mean_ms_per_frame"], 2)
        lines.append(f"{rid} & {fps} & {ov} & {vflag} & {eff} & {msf} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", required=True, type=str)
    ap.add_argument("--mouth-dir", required=True, type=str)
    ap.add_argument("--audio-dir", required=True, type=str)
    ap.add_argument("--texts-json", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--mouth-x", type=int, default=100)
    ap.add_argument("--mouth-y", type=int, default=300)
    ap.add_argument("--resolution", type=str, default="", help='Optional "WxH" e.g. 512x512')
    ap.add_argument("--max-clips", type=int, default=10)
    ap.add_argument("--no-video", action="store_true", help="Skip video encoding to speed ablations")
    args = ap.parse_args()

    face_path = Path(args.face)
    mouth_dir = Path(args.mouth_dir)
    audio_dir = Path(args.audio_dir)
    texts_json = Path(args.texts_json)
    out_dir = Path(args.out_dir)
    videos_root = out_dir / "videos"

    out_dir.mkdir(parents=True, exist_ok=True)
    videos_root.mkdir(parents=True, exist_ok=True)

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

    # Ablation grid (you can extend this)
    runs = [
        {"run_id": "A0_linear_fps30_ov0.5", "fps": 30, "overlap": 0.5, "use_vedic": False, "vedic_strength": 0.0},
        {"run_id": "A1_vedic_fps30_ov0.5", "fps": 30, "overlap": 0.5, "use_vedic": True,  "vedic_strength": 0.15},
        {"run_id": "A2_vedic_fps30_ov0.2", "fps": 30, "overlap": 0.2, "use_vedic": True,  "vedic_strength": 0.15},
        {"run_id": "A3_vedic_fps30_ov0.8", "fps": 30, "overlap": 0.8, "use_vedic": True,  "vedic_strength": 0.15},
        {"run_id": "A4_vedic_fps25_ov0.5", "fps": 25, "overlap": 0.5, "use_vedic": True,  "vedic_strength": 0.15},
    ]

    rows: List[Dict[str, object]] = []

    for run in runs:
        run_id = str(run["run_id"])
        fps = int(run["fps"])
        overlap = float(run["overlap"])
        use_vedic = bool(run["use_vedic"])
        vedic_strength = float(run["vedic_strength"])

        print(f"\n===== RUN {run_id} =====")
        run_videos = videos_root / run_id
        run_videos.mkdir(parents=True, exist_ok=True)

        for clip_i, (wav_name, text) in enumerate(items, start=1):
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
                overlap=overlap,
                use_vedic=use_vedic,
                vedic_strength=vedic_strength,
            )

            total_frames = int(math.ceil(audio_s * fps))
            out_video = run_videos / f"{safe_slug(Path(wav_name).stem)}.mp4"

            if args.no_video:
                _, prof = profile_call(lambda: None)
            else:
                _, prof = profile_call(
                    render_viseme_sequence,
                    viseme_seq=vis_seq,
                    face_image_path=str(face_path),
                    mouth_sprites_dir=str(mouth_dir),
                    out_video_path=str(out_video),
                    fps=fps,
                    # mouth_position=(args.mouth_x, args.mouth_y),
                    resolution=resolution,
                    audio_path=str(wav_path),
                )

            eff_fps = (total_frames / prof.wall_s) if prof.wall_s > 0 else 0.0
            ms_per_frame = (1000.0 * prof.wall_s / total_frames) if total_frames > 0 else 0.0

            row = {
                "run_id": run_id,
                "clip_index": clip_i,
                "wav": wav_name,
                "audio_s": round(audio_s, 6),
                "n_phonemes": len(phonemes),
                "n_viseme_segments": len(vis_seq),
                "fps": fps,
                "overlap": overlap,
                "use_vedic": int(use_vedic),
                "vedic_strength": vedic_strength if use_vedic else 0.0,
                "wall_s": round(prof.wall_s, 6),
                "cpu_pct": round(prof.cpu_pct, 6) if isinstance(prof.cpu_pct, float) else prof.cpu_pct,
                "mem_delta_mb": round(prof.mem_delta_mb, 6) if isinstance(prof.mem_delta_mb, float) else prof.mem_delta_mb,
                "total_frames": total_frames,
                "eff_fps": round(eff_fps, 6),
                "ms_per_frame": round(ms_per_frame, 6),
                "out_video": "" if args.no_video else str(out_video),
            }
            rows.append(row)
            print(f"[{run_id}] {clip_i}/{len(items)} {wav_name}: wall={row['wall_s']}s  eff_fps={row['eff_fps']}")

    # Write outputs
    per_run_csv = out_dir / "per_run.csv"
    summary_csv = out_dir / "summary.csv"
    table_tex = out_dir / "table.tex"

    if not rows:
        print("[error] No rows produced.")
        return

    fieldnames = list(rows[0].keys())
    write_csv(per_run_csv, rows, fieldnames=fieldnames)

    summaries = summarize_runs(rows)
    if summaries:
        write_csv(summary_csv, summaries, fieldnames=list(summaries[0].keys()))
        write_latex_table(table_tex, summaries)

    print("\nWritten:")
    print(f"  - {per_run_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {table_tex}")


if __name__ == "__main__":
    main()

