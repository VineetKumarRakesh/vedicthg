#!/usr/bin/env python3
# experiments/ablation/no_coarticulation.py

import os
import glob
import csv
import argparse
import tempfile
from vedicthg.audio_processing import extract_mfcc
from vedicthg.phoneme_align import recognize_phonemes, align_phonemes_with_transcript
from vedicthg.viseme_mapper import phoneme_to_viseme_sequence
from vedicthg.render_engine import render_viseme_sequence
from vedicthg.evaluation import profile_function, measure_lip_sync_accuracy

def ablate_no_coarticulation(
    audio_dir: str,
    face_path: str,
    mouth_dir: str,
    output_csv: str,
    fps: int
):
    """
    Run the full pipeline with coarticulation disabled (overlap=0)
    for each WAV in audio_dir, profile rendering, and measure lip-sync accuracy.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    header = [
        'audio_file',
        'lip_sync_accuracy_%',
        'lip_sync_avg_error_ms',
        'render_latency_s',
        'cpu_percent',
        'mem_delta_mb'
    ]
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        wav_paths = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
        for wav_path in wav_paths:
            print(f'Ablation (no coarticulation) on {wav_path}')
            # 1) Recognize phonemes
            phonemes, times = recognize_phonemes(wav_path)

            # 2) Map phonemes to viseme sequence with overlap=0
            vis_seq = phoneme_to_viseme_sequence(phonemes, times, overlap=0.0)

            # 3) Profile rendering to a temp file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_path = tmp.name
            render_stats = profile_function(
                render_viseme_sequence,
                vis_seq,
                face_path,
                mouth_dir,
                tmp_path,
                fps
            )
            # Clean up temp video
            try:
                os.remove(tmp_path)
            except OSError:
                pass

            # 4) Measure lip-sync accuracy
            accuracy, avg_error = measure_lip_sync_accuracy(phonemes, times, vis_seq)

            # 5) Write metrics
            _, latency, cpu_pct, mem_mb = render_stats
            row = [
                os.path.basename(wav_path),
                f'{accuracy:.2f}',
                f'{avg_error:.2f}',
                f'{latency:.3f}',
                f'{cpu_pct:.1f}',
                f'{mem_mb:.2f}'
            ]
            writer.writerow(row)
            print(f'  Accuracy: {accuracy:.2f}%, Latency: {latency:.3f}s, CPU: {cpu_pct:.1f}%, Mem Î”: {mem_mb:.2f}MB\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ablation: no coarticulation (overlap=0) on audio files.'
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
        '--mouth-dir', type=str, required=True,
        help='Directory with mouth-sprite PNGs named 0.png...14.png.'
    )
    parser.add_argument(
        '--output-csv', type=str,
        default='experiments/ablation/results/no_coarticulation.csv',
        help='Path to write the ablation CSV results.'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Frame rate for rendering.'
    )

    args = parser.parse_args()
    ablate_no_coarticulation(
        audio_dir=args.audio_dir,
        face_path=args.face,
        mouth_dir=args.mouth_dir,
        output_csv=args.output_csv,
        fps=args.fps
    )

