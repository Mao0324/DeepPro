#!/usr/bin/env python3
"""Build challenge centroid TXT files from the best ensemble sweep result."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from probability_ensemble_sweep import read_probability
from tools_forSatVideoIRSTD.seg2centroid_txt import (
    calculate_centroids,
    format_centroid_line,
)


def load_winner(paths):
    candidates = []
    for path in paths:
        with path.open(encoding='utf-8') as input_file:
            payload = json.load(input_file)
        candidates.append((payload['best']['f1'], path, payload))
    _, path, payload = max(
        candidates,
        key=lambda item: (
            item[0], item[2]['best']['recall'],
            -item[2]['best']['false_positive'],
        ),
    )
    return path, payload


def filtered_binary(probability, threshold, minimum_area):
    binary = np.asarray(probability > threshold, dtype=np.uint8)
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    keep = np.zeros(component_count, dtype=bool)
    if component_count > 1:
        keep[1:] = statistics[1:, cv2.CC_STAT_AREA] >= minimum_area
    return np.asarray(keep[labels], dtype=np.uint8) * 255


def main():
    parser = argparse.ArgumentParser(
        description='Select the best sweep JSON and emit challenge centroid TXT.'
    )
    parser.add_argument('--sweep-json', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()

    sweep_paths = [path.expanduser().resolve() for path in args.sweep_json]
    for path in sweep_paths:
        if not path.is_file():
            parser.error('sweep JSON does not exist: %s' % path)
    selected_path, payload = load_winner(sweep_paths)
    best = payload['best']
    root_a = Path(payload['prediction_root_a'])
    root_b = Path(payload['prediction_root_b'])
    weight_a = float(best['weight_a'])
    threshold = float(best['threshold'])
    minimum_area = int(best['min_area'])

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error('output directory is not empty: %s' % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences_a = {path.name: path for path in root_a.iterdir() if path.is_dir()}
    sequences_b = {path.name: path for path in root_b.iterdir() if path.is_dir()}
    if sequences_a.keys() != sequences_b.keys():
        raise ValueError('ensemble prediction sequence sets differ')

    for sequence_index, name in enumerate(sorted(sequences_a), start=1):
        files_a = sorted(sequences_a[name].glob('*.png'))
        files_b = sorted(sequences_b[name].glob('*.png'))
        if len(files_a) != len(files_b):
            raise ValueError('%s frame count differs between models' % name)
        output_path = output_dir / ('%s.txt' % name)
        with output_path.open('w', encoding='utf-8') as output_file:
            for frame_index, (path_a, path_b) in enumerate(
                    zip(files_a, files_b), start=1):
                probability = (
                    weight_a * read_probability(path_a)
                    + (1.0 - weight_a) * read_probability(path_b)
                )
                binary = filtered_binary(
                    probability, threshold, minimum_area
                )
                output_file.write(
                    format_centroid_line(
                        frame_index, calculate_centroids(binary)
                    ) + '\n'
                )
        if sequence_index % 25 == 0 or sequence_index == len(sequences_a):
            print(
                'wrote %d/%d sequence TXT files'
                % (sequence_index, len(sequences_a)),
                flush=True,
            )

    manifest = {
        'selected_sweep': str(selected_path),
        'prediction_root_a': str(root_a),
        'prediction_root_b': str(root_b),
        'weight_a': weight_a,
        'weight_b': 1.0 - weight_a,
        'threshold': threshold,
        'min_area': minimum_area,
        'proxy_precision': best['precision'],
        'proxy_recall': best['recall'],
        'proxy_f1': best['f1'],
        'centroid_dir': str(output_dir),
    }
    manifest_path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('w', encoding='utf-8') as output_file:
        json.dump(manifest, output_file, indent=2, ensure_ascii=False)
        output_file.write('\n')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
