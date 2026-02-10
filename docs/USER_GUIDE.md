# User Guide

VedicTHG is designed to be **dataset-free**: the repository does not ship any face images, videos, or copyrighted datasets.
You either **generate synthetic dummy assets** (for a safe quick test) or **bring your own assets** (with consent).

## Install

```bash
python -m pip install --upgrade pip
pip install -e .
```

If you want pinned versions:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick demo (safe, synthetic assets)

1) Generate dummy assets:

```bash
python scripts/prepare_dummy_assets.py
```

2) Render a demo clip:

```bash
python scripts/run_demo.py
```

Output: `results/demo.mp4`

## Using your own assets

Place your inputs under `data/raw/` (this folder is gitignored):

- `data/raw/avatar.png` — a face image (front-facing works best)
- `data/raw/visemes/0.png ... 14.png` — mouth sprites
- `data/raw/example.wav` — audio (16 kHz mono WAV recommended)

Then run:

```bash
python -m vedicthg.demo --audio data/raw/example.wav --face data/raw/avatar.png --mouth-dir data/raw/visemes --output results/my_video.mp4 --fps 30
```

## Benchmarking

Benchmark scripts live in `experiments/benchmarks/`. A typical run:

```bash
python experiments/benchmarks/run_benchmark.py --root . --num-clips 10 --fps 30 --render
```

Outputs (gitignored) are written under:
- `experiments/benchmarks/results/`
- `results/`

## Notes

- **FFmpeg:** MoviePy may require FFmpeg installed (especially on Windows).
- **MediaPipe:** For the FaceMesh-based rig, Python 3.10/3.11 is recommended on Windows.
- **Optional Whisper script:** if you use `scripts/extract_phonemes.py`, install extras:
  ```bash
  pip install -e ".[whisper]"
  ```
