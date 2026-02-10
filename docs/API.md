# API Reference

This document provides a comprehensive overview of the public interfaces in the `vedicthg` package (`src/vedicthg/`).

---

## 1. Module: `vedicthg.audio_processing`

```python
load_audio(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]
```

* **Description:** Load a WAV audio file as a mono signal at the specified sampling rate.
* **Returns:** `(signal, sample_rate)` where `signal` is a normalized NumPy array and `sample_rate` is the sampling rate.

```python
pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray
```

* **Description:** Apply a pre-emphasis filter to boost high-frequency components.

```python
extract_mfcc(
    audio_path: str,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: Optional[int] = None,
    pre_emph_coeff: float = 0.97
) -> Tuple[np.ndarray, int]
```

* **Description:** Load audio, apply pre-emphasis, extract MFCCs and their deltas.
* **Returns:** `(mfcc_all, sample_rate)`, where `mfcc_all` has shape `(2*n_mfcc, T)`.

---

## 2. Module: `vedicthg.phoneme_align`

```python
recognize_phonemes(
    audio_path: str,
    hmm_dir: Optional[str] = None,
    dict_path: Optional[str] = None,
    phone_lm_path: Optional[str] = None
) -> Tuple[List[str], List[Tuple[float, float]]]
```

* **Description:** Perform phoneme-level recognition (all-phone) using PocketSphinx.
* **Returns:** `phonemes` list and `times` list of `(start, end)` in seconds.

```python
align_phonemes_with_transcript(
    audio_path: str,
    transcript: str,
    dict_path: Optional[str] = None,
    hmm_dir: Optional[str] = None
) -> Tuple[List[str], List[Tuple[float, float]]]
```

* **Description:** Force-align a known transcript to the audio using a JSGF grammar.

---

## 3. Module: `vedicthg.viseme_mapper`

```python
blend_visemes(vi: int, vj: int, alpha: float) -> np.ndarray
```

* **Description:** Blend two visemes with overlap fraction `alpha`, using a Vedic-inspired crosswise term.
* **Returns:** A 2-element parameter vector for the blended viseme.

```python
phoneme_to_viseme_sequence(
    phonemes: List[str],
    times: List[Tuple[float, float]],
    overlap: float = 0.5
) -> List[Tuple[int, float, float, np.ndarray]]
```

* **Description:** Convert phoneme list and timestamps into a timed viseme sequence with coarticulation.
* **Returns:** A list of `(viseme_idx, start, end, params)` tuples.

---

## 4. Module: `vedicthg.render_engine`

```python
render_viseme_sequence(
    viseme_seq: List[Tuple[int, float, float, np.ndarray]],
    face_image_path: str,
    mouth_sprites_dir: str,
    out_video_path: str,
    fps: int = 30,
    mouth_position: Tuple[int, int] = (100, 300),
    resolution: Optional[Tuple[int, int]] = None
) -> None
```

* **Description:** Render a sequence of visemes onto a 2D avatar and write out an MP4 video.

---

## 5. Module: `vedicthg.evaluation`

```python
profile_function(func: Callable, *args, **kwargs) -> Tuple[Any, float, float, float]
```

* **Description:** Profile a function call, returning `(result, latency_s, cpu_percent, mem_delta_mb)`.

```python
measure_lip_sync_accuracy(
    phonemes: List[str],
    phoneme_times: List[Tuple[float, float]],
    viseme_seq: List[Tuple[int, float, float, Any]],
    tolerance_ms: float = 40.0
) -> Tuple[float, float]
```

* **Description:** Compute lip-sync accuracy (%) and average timing error (ms).

```python
log_performance_metrics(
    phonemes: List[str],
    phoneme_times: List[Tuple[float, float]],
    viseme_seq: List[Tuple[int, float, float, Any]],
    render_stats: Tuple[Any, float, float, float],
    output_csv: Optional[str] = None
) -> Dict[str, float]
```

* **Description:** Log metrics to console and optionally append to a CSV file.

---

## 6. Module: `vedicthg.utils`

```python
ensure_dir(path: str) -> None
```

* **Description:** Create a directory (and parents) if it doesn’t exist.

```python
download_file(url: str, dest_path: str, chunk_size: int = 1024) -> None
```

* **Description:** Download a file via HTTP streaming.

```python
extract_zip(zip_path: str, extract_to: str) -> None
extract_tar(tar_path: str, extract_to: str) -> None
```

* **Description:** Extract ZIP or TAR archives.

```python
download_and_extract(url: str, output_dir: str, archive_name: str) -> None
```

* **Description:** Download an archive and extract it in one step.

```python
download_data() -> None
```

* **Description:** Fetch and prepare all required datasets and pretrained models.

---

## 7. Module: `vedicthg.demo`

```python
main() -> None
```

* **Description:** CLI entry point for end-to-end demo. Accepts arguments for audio path, face image, mouth sprites, output video, FPS, overlap, and `--download-data` flag.

**Usage Example:**

```bash
python vedicthg.demo.py --download-data --audio data/raw/example.wav \
    --face data/raw/avatar.png --mouth-dir data/raw/visemes \
    --output results/demo.mp4 --fps 30 --overlap 0.5
```
