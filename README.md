# VedicTHG

**VedicTHG** is a lightweight, *bring-your-own-assets* toolkit for **audio-driven talking-head generation**.
It turns an input audio clip into a sequence of visemes (mouth shapes) and renders a lip-synced video by
compositing mouth sprites onto a reference face image (optionally with simple motion/rigging).

This repository is **GitHub-ready** and intentionally ships **no datasets** and **no personal face media**.
You provide your own assets (with consent) or generate the included dummy assets for testing.

Project page: https://vineetkumarrakesh.github.io/vedicthg/

## What’s included

- **Audio → phonemes** via PocketSphinx (allphone/phoneme decoding)
- **Phoneme → viseme mapping** with optional overlap/coarticulation heuristics (“Vedic” smoothing)
- **Rendering engine** that overlays viseme sprites onto a face image (OpenCV + MoviePy)
- **Optional dynamic rig** using MediaPipe FaceMesh to stabilize mouth placement
- **Benchmarking scripts** (speed, CPU/memory, simple sync proxies) and ablations

## Method summary (high level)

1. **Audio processing**: load audio, optional MFCC extraction.
2. **Phoneme alignment**: decode an approximate phoneme sequence and timings.
3. **Viseme synthesis**: map phonemes to discrete viseme IDs and blend with overlap for smoothness.
4. **Rendering**: for each frame, select/blend a mouth sprite, warp/compose it into the mouth region,
   and write the video with the original audio.

> Note: This repo provides a reproducible **engineering pipeline**. It does **not** include or redistribute
> any copyrighted datasets (GRID/LRS/VoxCeleb/etc.).

## Install

**Recommended:** Python 3.10 or 3.11 (Windows/Linux).  
(Some wheels like `mediapipe` may be limited on newer Python versions.)

```bash
# from the repo root
python -m pip install --upgrade pip
pip install -e .
```

If you prefer a pinned environment:

```bash
pip install -r requirements.txt
pip install -e .
```

### Optional extras

- Whisper-based phoneme extraction script:
  ```bash
  pip install -e ".[whisper]"
  ```
- Some analysis/ablation scripts (identity metrics):
  ```bash
  pip install -e ".[analysis]"
  ```

## Quickstart demo (no datasets)

1) Generate dummy assets (safe, synthetic):

```bash
python scripts/prepare_dummy_assets.py
```

This creates:
- `data/raw/avatar.png` (a simple synthetic face)
- `data/raw/visemes/0.png ... 14.png` (sprite mouth shapes)
- `data/raw/benchmark/clip_01.wav ...` (synthetic audio clips)

2) Run the simple demo renderer:

```bash
python scripts/run_demo.py
```

Output:
- `results/demo.mp4`

If you want the full CLI demo (phoneme recognition + profiling):

```bash
python -m vedicthg.demo --audio data/raw/benchmark/clip_01.wav --face data/raw/avatar.png --mouth-dir data/raw/visemes --output results/demo_cli.mp4 --fps 30
```

## Bring your own assets

**You must have permission/consent for any face media you use.**

Minimum inputs:

- **Face image**: `data/raw/avatar.png`  
  A front-facing portrait works best.
- **Mouth sprites folder**: `data/raw/visemes/`  
  PNG sprites named `0.png ... 14.png` corresponding to the viseme IDs used by the mapper.
- **Audio**: `data/raw/example.wav` (16 kHz mono recommended)

Recommended workflow:

1. Put your assets under `data/raw/` (this folder is gitignored).
2. Run:
   ```bash
   python -m vedicthg.demo --audio data/raw/example.wav --face data/raw/avatar.png --mouth-dir data/raw/visemes --output results/my_run.mp4
   ```

### Mouth sprite conventions

- Transparent background PNGs are recommended.
- Keep a consistent crop/scale across all sprites.
- Viseme indices are defined in `src/vedicthg/viseme_mapper.py`.

## Benchmarking

Benchmarks live in `experiments/benchmarks/`.

After generating or adding benchmark clips in `data/raw/benchmark/`:

```bash
python experiments/benchmarks/run_benchmark.py --root . --num-clips 10 --fps 30 --render
```

Typical outputs (gitignored):
- `experiments/benchmarks/results/*.csv`
- `results/*.mp4` (if rendering is enabled)

## Repository layout

```
VedicTHG/
  src/vedicthg/        # installable python package
  scripts/             # runnable scripts (no PYTHONPATH hacks needed)
  experiments/         # benchmarks, ablations, analysis
  docs/                # extra documentation
  data/                # empty placeholders (gitignored; BYO assets)
  results/             # empty placeholders (gitignored)
  assets/              # optional project media (kept empty here)
```

## Troubleshooting

- **`mediapipe` install issues (Windows):** use Python 3.10/3.11 and upgrade pip.
- **MoviePy / FFmpeg errors:** ensure FFmpeg is installed and available on PATH.
- **No audio / wrong codec:** convert audio to 16 kHz mono WAV.
- **Mouth misalignment:** try enabling rigging options in the renderer and use a cleaner, front-facing portrait.
- **PocketSphinx model files:** PocketSphinx includes default models; if you override paths, verify them.

## Responsible use

This project can create realistic-looking talking head videos when paired with suitable assets.
Please use it responsibly:

- Use only face media you own or have explicit permission to use.
- Clearly label synthetic media when shared publicly.
- Do not use for impersonation, deception, harassment, or political manipulation.

## Citation

If you use VedicTHG in academic work, please cite:

```bibtex
@software{rakesh_vedicthg_2026,
  author  = {Vineet Kumar Rakesh},
  title   = {VedicTHG: Bring-your-own-assets audio-driven talking-head generation toolkit},
  year    = {2026},
  version = {0.1.0},
  url     = {https://vineetkumarrakesh.github.io/vedicthg/}
}
```

See also: `CITATION.cff`.

---
**No datasets included. Bring your own assets.**
