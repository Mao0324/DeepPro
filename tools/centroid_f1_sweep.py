#!/usr/bin/env python3
"""Sweep probability thresholds using competition-oriented centroid F1.

The prediction root must contain one directory per validation sequence and one
grayscale PNG probability map per frame. Frames are paired by sorted order
because test.py deliberately renumbers challenge outputs from one.
"""

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def parse_float_grid(specification):
    """Parse comma values or inclusive start:stop:step notation."""
    if ':' not in specification:
        values = [float(item) for item in specification.split(',') if item]
    else:
        fields = specification.split(':')
        if len(fields) != 3:
            raise ValueError('threshold grid must be start:stop:step')
        start, stop, step = map(float, fields)
        if step <= 0 or stop < start:
            raise ValueError('threshold grid requires stop >= start and step > 0')
        count = int(math.floor((stop - start) / step + 1.0e-9)) + 1
        values = [start + index * step for index in range(count)]
        if values[-1] < stop - 1.0e-9:
            values.append(stop)
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError('thresholds must be in [0, 1]')
    return tuple(sorted(set(round(value, 8) for value in values)))


def parse_positive_ints(specification):
    values = tuple(sorted(set(
        int(item) for item in specification.split(',') if item
    )))
    if not values or any(value < 1 for value in values):
        raise ValueError('minimum areas must be positive integers')
    return values


def read_probability(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('cannot read prediction: %s' % path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.issubdtype(image.dtype, np.integer):
        scale = float(np.iinfo(image.dtype).max)
    else:
        scale = 1.0 if (image.size == 0 or float(image.max()) <= 1.0) else 255.0
    return image.astype(np.float32) / scale


def component_geometry(binary):
    _, _, statistics, centroids = cv2.connectedComponentsWithStats(
        np.asarray(binary, dtype=np.uint8), connectivity=8
    )
    return statistics[1:, cv2.CC_STAT_AREA], centroids[1:]


def matched_count(predicted, target, distance):
    """Return maximum valid one-to-one matches inside the distance gate."""
    if len(predicted) == 0 or len(target) == 0:
        return 0
    pairwise = np.linalg.norm(
        predicted[:, None, :] - target[None, :, :], axis=2
    )
    adjacency = csr_matrix(pairwise <= distance)
    matches = maximum_bipartite_matching(adjacency, perm_type='column')
    return int(np.count_nonzero(matches >= 0))


def evaluate_sequence(task):
    (
        sequence_name,
        prediction_files,
        target_files,
        thresholds,
        minimum_areas,
        match_distance,
    ) = task
    if len(prediction_files) != len(target_files):
        raise ValueError(
            '%s has %d predictions but %d labels'
            % (sequence_name, len(prediction_files), len(target_files))
        )
    counts = np.zeros(
        (len(thresholds), len(minimum_areas), 3), dtype=np.int64
    )
    for prediction_path, target_path in zip(prediction_files, target_files):
        probability = read_probability(prediction_path)
        target_mask = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        if target_mask is None:
            raise ValueError('cannot read target: %s' % target_path)
        if probability.shape != target_mask.shape:
            raise ValueError(
                'shape mismatch in %s: %s versus %s'
                % (sequence_name, probability.shape, target_mask.shape)
            )
        _, target_centroids = component_geometry(target_mask > 0)
        target_count = len(target_centroids)
        for threshold_index, threshold in enumerate(thresholds):
            areas, predicted_centroids = component_geometry(
                probability > threshold
            )
            for area_index, minimum_area in enumerate(minimum_areas):
                selected = predicted_centroids[areas >= minimum_area]
                true_positive = matched_count(
                    selected, target_centroids, match_distance
                )
                counts[threshold_index, area_index] += (
                    true_positive,
                    len(selected) - true_positive,
                    target_count - true_positive,
                )
    return sequence_name, counts


def discover_tasks(
    prediction_root,
    target_root,
    thresholds,
    minimum_areas,
    match_distance,
):
    prediction_sequences = sorted(
        path for path in prediction_root.iterdir() if path.is_dir()
    )
    if not prediction_sequences:
        raise ValueError('no prediction sequence directories in %s' % prediction_root)
    target_names = {
        path.name for path in target_root.iterdir() if path.is_dir()
    }
    prediction_names = {path.name for path in prediction_sequences}
    if prediction_names != target_names:
        missing = sorted(target_names - prediction_names)
        extra = sorted(prediction_names - target_names)
        raise ValueError(
            'sequence mismatch; missing=%s extra=%s'
            % (missing[:10], extra[:10])
        )
    tasks = []
    for prediction_sequence in prediction_sequences:
        target_sequence = target_root / prediction_sequence.name / 'mask'
        prediction_files = tuple(sorted(prediction_sequence.glob('*.png')))
        target_files = tuple(sorted(
            path for path in target_sequence.iterdir() if path.is_file()
        ))
        tasks.append((
            prediction_sequence.name,
            prediction_files,
            target_files,
            thresholds,
            minimum_areas,
            match_distance,
        ))
    return tasks


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def build_rows(counts, thresholds, minimum_areas):
    rows = []
    for threshold_index, threshold in enumerate(thresholds):
        for area_index, minimum_area in enumerate(minimum_areas):
            true_positive, false_positive, false_negative = (
                int(value) for value in counts[threshold_index, area_index]
            )
            precision = safe_divide(
                true_positive, true_positive + false_positive
            )
            recall = safe_divide(
                true_positive, true_positive + false_negative
            )
            f1 = safe_divide(2.0 * precision * recall, precision + recall)
            rows.append({
                'threshold': threshold,
                'min_area': minimum_area,
                'true_positive': true_positive,
                'false_positive': false_positive,
                'false_negative': false_negative,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Sweep probability threshold/minimum area using centroid F1.'
    )
    parser.add_argument('--prediction-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--thresholds', default='0.25:0.60:0.01')
    parser.add_argument('--min-areas', default='1,2')
    parser.add_argument('--match-distance', type=float, default=2.0)
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--output-csv', type=Path)
    parser.add_argument('--output-json', type=Path)
    args = parser.parse_args()

    prediction_root = args.prediction_root.expanduser().resolve()
    target_root = args.data_root.expanduser().resolve() / args.split
    if not prediction_root.is_dir():
        parser.error('prediction root does not exist: %s' % prediction_root)
    if not target_root.is_dir():
        parser.error('target split does not exist: %s' % target_root)
    if args.match_distance < 0:
        parser.error('--match-distance must be non-negative')
    if args.workers < 1:
        parser.error('--workers must be positive')

    try:
        thresholds = parse_float_grid(args.thresholds)
        minimum_areas = parse_positive_ints(args.min_areas)
    except ValueError as error:
        parser.error(str(error))

    tasks = discover_tasks(
        prediction_root,
        target_root,
        thresholds,
        minimum_areas,
        args.match_distance,
    )
    total_counts = np.zeros(
        (len(thresholds), len(minimum_areas), 3), dtype=np.int64
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for sequence_index, (_, sequence_counts) in enumerate(
            executor.map(evaluate_sequence, tasks), start=1
        ):
            total_counts += sequence_counts
            if sequence_index % 25 == 0 or sequence_index == len(tasks):
                print(
                    'evaluated %d/%d sequences'
                    % (sequence_index, len(tasks)),
                    flush=True,
                )

    rows = build_rows(total_counts, thresholds, minimum_areas)
    rows.sort(key=lambda row: (row['threshold'], row['min_area']))
    best = max(
        rows,
        key=lambda row: (
            row['f1'], row['recall'], -row['false_positive']
        ),
    )
    output_csv = args.output_csv or prediction_root / 'centroid_f1_sweep.csv'
    output_json = args.output_json or prediction_root / 'centroid_f1_sweep.json'
    write_csv(output_csv, rows)
    payload = {
        'prediction_root': str(prediction_root),
        'target_root': str(target_root),
        'match_distance': args.match_distance,
        'sequence_count': len(tasks),
        'best': best,
        'rows': rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open('w', encoding='utf-8') as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write('\n')

    print(
        'BEST threshold={threshold:.4f} min_area={min_area} '
        'precision={precision:.6f} recall={recall:.6f} f1={f1:.6f} '
        'TP={true_positive} FP={false_positive} FN={false_negative}'.format(
            **best
        )
    )
    print('CSV: %s' % output_csv)
    print('JSON: %s' % output_json)


if __name__ == '__main__':
    main()
