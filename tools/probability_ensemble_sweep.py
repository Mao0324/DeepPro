#!/usr/bin/env python3
"""Sweep a two-model probability ensemble with centroid-level F1."""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from centroid_f1_sweep import (
    build_rows,
    component_geometry,
    matched_count,
    parse_float_grid,
    parse_positive_ints,
    read_probability,
)


def discover_tasks(root_a, root_b, target_root, weights, thresholds,
                   minimum_areas, match_distance):
    names_a = {path.name for path in root_a.iterdir() if path.is_dir()}
    names_b = {path.name for path in root_b.iterdir() if path.is_dir()}
    names_target = {path.name for path in target_root.iterdir() if path.is_dir()}
    if names_a != names_b or names_a != names_target:
        raise ValueError(
            'sequence mismatch: a=%d b=%d target=%d'
            % (len(names_a), len(names_b), len(names_target))
        )

    tasks = []
    for name in sorted(names_a):
        files_a = tuple(sorted((root_a / name).glob('*.png')))
        files_b = tuple(sorted((root_b / name).glob('*.png')))
        target_files = tuple(sorted(
            path for path in (target_root / name / 'mask').iterdir()
            if path.is_file()
        ))
        if not (len(files_a) == len(files_b) == len(target_files)):
            raise ValueError(
                '%s frame mismatch: a=%d b=%d target=%d'
                % (name, len(files_a), len(files_b), len(target_files))
            )
        tasks.append((
            name, files_a, files_b, target_files, weights, thresholds,
            minimum_areas, match_distance,
        ))
    return tasks


def evaluate_sequence(task):
    (name, files_a, files_b, target_files, weights, thresholds,
     minimum_areas, match_distance) = task
    counts = np.zeros(
        (len(weights), len(thresholds), len(minimum_areas), 3),
        dtype=np.int64,
    )
    for path_a, path_b, target_path in zip(files_a, files_b, target_files):
        probability_a = read_probability(path_a)
        probability_b = read_probability(path_b)
        target_mask = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        if target_mask is None:
            raise ValueError('cannot read target: %s' % target_path)
        if probability_a.shape != probability_b.shape \
                or probability_a.shape != target_mask.shape:
            raise ValueError('shape mismatch in %s' % name)

        _, target_centroids = component_geometry(target_mask > 0)
        target_count = len(target_centroids)
        for weight_index, weight_a in enumerate(weights):
            probability = (
                weight_a * probability_a
                + (1.0 - weight_a) * probability_b
            )
            for threshold_index, threshold in enumerate(thresholds):
                areas, predicted_centroids = component_geometry(
                    probability > threshold
                )
                for area_index, minimum_area in enumerate(minimum_areas):
                    selected = predicted_centroids[areas >= minimum_area]
                    true_positive = matched_count(
                        selected, target_centroids, match_distance
                    )
                    counts[weight_index, threshold_index, area_index] += (
                        true_positive,
                        len(selected) - true_positive,
                        target_count - true_positive,
                    )
    return name, counts


def rows_from_counts(counts, weights, thresholds, minimum_areas):
    rows = []
    for weight_index, weight_a in enumerate(weights):
        weight_rows = build_rows(
            counts[weight_index], thresholds, minimum_areas
        )
        for row in weight_rows:
            rows.append({'weight_a': weight_a, **row})
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Sweep A*w + B*(1-w) using centroid-level F1.'
    )
    parser.add_argument('--prediction-root-a', type=Path, required=True)
    parser.add_argument('--prediction-root-b', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--weights-a', default='0.5,0.65,0.8,1.0')
    parser.add_argument('--thresholds', default='0.35:0.55:0.025')
    parser.add_argument('--min-areas', default='1,2')
    parser.add_argument('--match-distance', type=float, default=2.0)
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()

    root_a = args.prediction_root_a.expanduser().resolve()
    root_b = args.prediction_root_b.expanduser().resolve()
    target_root = args.data_root.expanduser().resolve() / args.split
    for label, path in (('A', root_a), ('B', root_b), ('target', target_root)):
        if not path.is_dir():
            parser.error('%s root does not exist: %s' % (label, path))
    if args.match_distance < 0:
        parser.error('--match-distance must be non-negative')
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        weights = parse_float_grid(args.weights_a)
        thresholds = parse_float_grid(args.thresholds)
        minimum_areas = parse_positive_ints(args.min_areas)
    except ValueError as error:
        parser.error(str(error))

    tasks = discover_tasks(
        root_a, root_b, target_root, weights, thresholds, minimum_areas,
        args.match_distance,
    )
    total = np.zeros(
        (len(weights), len(thresholds), len(minimum_areas), 3),
        dtype=np.int64,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (_, sequence_counts) in enumerate(
                executor.map(evaluate_sequence, tasks), start=1):
            total += sequence_counts
            if index % 25 == 0 or index == len(tasks):
                print('evaluated %d/%d sequences' % (index, len(tasks)),
                      flush=True)

    rows = rows_from_counts(total, weights, thresholds, minimum_areas)
    rows.sort(key=lambda row: (
        row['weight_a'], row['threshold'], row['min_area']
    ))
    best = max(rows, key=lambda row: (
        row['f1'], row['recall'], -row['false_positive']
    ))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        'prediction_root_a': str(root_a),
        'prediction_root_b': str(root_b),
        'target_root': str(target_root),
        'match_distance': args.match_distance,
        'sequence_count': len(tasks),
        'best': best,
        'rows': rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open('w', encoding='utf-8') as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write('\n')
    print(
        'BEST weight_a={weight_a:.4f} threshold={threshold:.4f} '
        'min_area={min_area} precision={precision:.6f} '
        'recall={recall:.6f} f1={f1:.6f} TP={true_positive} '
        'FP={false_positive} FN={false_negative}'.format(**best)
    )
    print('CSV: %s' % args.output_csv)
    print('JSON: %s' % args.output_json)


if __name__ == '__main__':
    main()
