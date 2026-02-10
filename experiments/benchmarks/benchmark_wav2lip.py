#!/usr/bin/env python3
# experiments/benchmarks/benchmark_wav2lip.py
'''
python experiments/benchmarks/benchmark_wav2lip.py \
  --audio-dir data/raw/benchmark \
  --face data/raw/avatar.png \
  --checkpoint models/wav2lip_cpu/wav2lip_gan.pth \
  --output-csv experiments/benchmarks/results/wav2lip_benchmark.csv \
  --fps 30

'''
import os
import glob
import csv
import argparse
import tempfile
import subprocess
from vedicthg.evaluation import profile_function

def run_wav2lip_inference(
    face_image: str,
    audio_path: str,
    checkpoint_path: str,
    outfile: str,
    fps: int
):
    """
    Run Wav2Lip inference (CPU) to generate a lip-synced video.
    Assumes the Wav2Lip inference script is available at:
      models/wav2lip_cpu/inference.py
    """
    cmd = [
        "python", os.path.join("models", "wav2lip_cpu", "inference.py"),
        "--checkpoint_path", checkpoint_path,
        "--face", face_image,
        "--audio", audio_path,
        "--outfile", outfile,
        "--fps", str(fps),
        "--cpu"  # flag to force CPU inference if supported by script
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def benchmark_wav2lip(
    audio_dir: str,
    face_image: str,
    checkpoint_path: str,
    output_csv: str,
    fps: int
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    header = [
        'audio_file',
        'render_latency_s',
        'cpu_percent',
        'mem_delta_mb'
    ]
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        wav_paths = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
        for wav_path in wav_paths:
            print(f'Benchmarking Wav2Lip on {wav_path}')
            # profile the inference call
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                out_vid = tmp.name

            # Profile the Wav2Lip inference
            _, latency, cpu_pct, mem_mb = profile_function(
                run_wav2lip_inference,
                face_image,
                wav_path,
                checkpoint_path,
                out_vid,
                fps
            )

            # Clean up the generated video
            try:
                os.remove(out_vid)
            except OSError:
                pass

            # Write metrics
            writer.writerow([
                os.path.basename(wav_path),
                f'{latency:.3f}',
                f'{cpu_pct:.1f}',
                f'{mem_mb:.2f}'
            ])
            print(f'  Latency: {latency:.3f}s, CPU: {cpu_pct:.1f}%, Mem Î”: {mem_mb:.2f}MB\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark Wav2Lip CPU-only inference on a set of audio files.'
    )
    parser.add_argument(
        '--audio-dir', type=str, default='data/raw/benchmark',
        help='Directory containing WAV files to process.'
    )
    parser.add_argument(
        '--face', type=str, required=True,
        help='Path to the base face PNG image.'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to the Wav2Lip .pth checkpoint.'
    )
    parser.add_argument(
        '--output-csv', type=str,
        default='experiments/benchmarks/results/wav2lip_benchmark.csv',
        help='Path to write the benchmark CSV results.'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Frame rate for inference output.'
    )

    args = parser.parse_args()
    benchmark_wav2lip(
        audio_dir=args.audio_dir,
        face_image=args.face,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        fps=args.fps
    )

