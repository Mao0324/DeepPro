#!/usr/bin/env python3
"""Retain the best single-model centroid submission during checkpoint sweeps."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.build_ensemble_submission import filtered_binary
from tools.centroid_f1_sweep import read_probability
from tools_forSatVideoIRSTD.seg2centroid_txt import (
    calculate_centroids,
    format_centroid_line,
)


def score_key(best):
    return (
        float(best['f1']),
        float(best['recall']),
        -int(best['false_positive']),
    )


def existing_key(manifest_path):
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    return (
        float(payload['proxy_f1']),
        float(payload['proxy_recall']),
        -int(payload['proxy_false_positive']),
    )


def build_centroids(prediction_root, output_dir, threshold, minimum_area):
    sequences = sorted(path for path in prediction_root.iterdir() if path.is_dir())
    if not sequences:
        raise ValueError('no prediction sequence directories in %s' % prediction_root)
    for sequence_index, sequence in enumerate(sequences, start=1):
        prediction_files = sorted(sequence.glob('*.png'))
        if not prediction_files:
            raise ValueError('no PNG predictions in %s' % sequence)
        output_path = output_dir / ('%s.txt' % sequence.name)
        with output_path.open('w', encoding='utf-8') as output_file:
            for frame_index, prediction_path in enumerate(
                prediction_files, start=1
            ):
                binary = filtered_binary(
                    read_probability(prediction_path), threshold, minimum_area
                )
                output_file.write(
                    format_centroid_line(
                        frame_index, calculate_centroids(binary)
                    ) + '\n'
                )
        if sequence_index % 25 == 0 or sequence_index == len(sequences):
            print(
                'wrote %d/%d sequence TXT files'
                % (sequence_index, len(sequences)),
                flush=True,
            )


def replace_directory(staged_dir, output_dir):
    backup_dir = output_dir.with_name(output_dir.name + '.previous')
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if output_dir.exists():
        os.replace(output_dir, backup_dir)
    try:
        os.replace(staged_dir, output_dir)
    except Exception:
        if backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()

    sweep_path = args.sweep_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if not sweep_path.is_file():
        parser.error('sweep JSON does not exist: %s' % sweep_path)

    payload = json.loads(sweep_path.read_text(encoding='utf-8'))
    best = payload['best']
    previous_key = existing_key(manifest_path)
    candidate_key = score_key(best)
    if previous_key is not None and candidate_key <= previous_key:
        print(
            'kept previous candidate: current proxy F1 %.6f is not better'
            % float(best['f1'])
        )
        return

    prediction_root = Path(payload['prediction_root']).expanduser().resolve()
    if not prediction_root.is_dir():
        raise FileNotFoundError(prediction_root)
    threshold = float(best['threshold'])
    minimum_area = int(best['min_area'])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(tempfile.mkdtemp(
        prefix=output_dir.name + '.staged-', dir=str(output_dir.parent)
    ))
    try:
        build_centroids(
            prediction_root, staged_dir, threshold, minimum_area
        )
        replace_directory(staged_dir, output_dir)
    finally:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)

    manifest = {
        'selected_sweep': str(sweep_path),
        'prediction_root': str(prediction_root),
        'threshold': threshold,
        'min_area': minimum_area,
        'proxy_precision': float(best['precision']),
        'proxy_recall': float(best['recall']),
        'proxy_f1': float(best['f1']),
        'proxy_true_positive': int(best['true_positive']),
        'proxy_false_positive': int(best['false_positive']),
        'proxy_false_negative': int(best['false_negative']),
        'centroid_dir': str(output_dir),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + '.tmp')
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    temporary_manifest.replace(manifest_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
