#!/usr/bin/env python3
"""Validate a flat SatVideoIRSDT trajectory submission archive."""

import argparse
import math
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split', default='val')
    args = parser.parse_args()

    archive_path = args.archive.expanduser().resolve()
    split_root = args.data_root.expanduser().resolve() / args.split
    if not archive_path.is_file():
        parser.error('archive does not exist: %s' % archive_path)
    if not split_root.is_dir():
        parser.error('dataset split does not exist: %s' % split_root)

    nested_image_root = split_root / 'img'
    sequence_root = (
        nested_image_root if nested_image_root.is_dir() else split_root
    )
    expected = {}
    for sequence in sequence_root.iterdir():
        if not sequence.is_dir():
            continue
        image_root = sequence if sequence_root == nested_image_root else sequence / 'img'
        image_files = tuple(sorted(
            path for path in image_root.iterdir()
            if path.is_file()
        ))
        if not image_files:
            raise ValueError('sequence has no images: %s' % sequence)
        with Image.open(image_files[0]) as image:
            width, height = image.size
        expected['%s.txt' % sequence.name] = {
            'frames': len(image_files),
            'width': width,
            'height': height,
        }
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('archive contains duplicate members')
        if any(
            len(PurePosixPath(name).parts) != 1
            or PurePosixPath(name).suffix != '.txt'
            for name in names
        ):
            raise ValueError('archive must contain only flat TXT members')
        if set(names) != set(expected):
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise ValueError(
                'sequence mismatch: missing=%s extra=%s'
                % (missing[:10], extra[:10])
            )
        total_frames = 0
        total_detections = 0
        for name in sorted(names):
            lines = archive.read(name).decode('utf-8').splitlines()
            sequence_specification = expected[name]
            if len(lines) != sequence_specification['frames']:
                raise ValueError(
                    '%s has %d lines; expected %d'
                    % (name, len(lines), sequence_specification['frames'])
                )
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                if len(fields) < 2:
                    raise ValueError('%s:%d is malformed' % (name, line_number))
                if int(fields[0]) != line_number:
                    raise ValueError(
                        '%s:%d frame index is not sequential'
                        % (name, line_number)
                    )
                count = int(fields[1])
                if count < 0 or len(fields) != 2 + 3 * count:
                    raise ValueError(
                        '%s:%d has an invalid target count'
                        % (name, line_number)
                    )
                identifiers = []
                for index in range(count):
                    identifier = int(fields[2 + 3 * index])
                    x_coordinate = float(fields[3 + 3 * index])
                    y_coordinate = float(fields[4 + 3 * index])
                    if identifier < 0:
                        raise ValueError('%s:%d has a negative track ID' % (
                            name, line_number
                        ))
                    if not (
                        math.isfinite(x_coordinate)
                        and math.isfinite(y_coordinate)
                        and 0.0 <= x_coordinate < sequence_specification['width']
                        and 0.0 <= y_coordinate < sequence_specification['height']
                    ):
                        raise ValueError('%s:%d has invalid coordinates' % (
                            name, line_number
                        ))
                    identifiers.append(identifier)
                if len(identifiers) != len(set(identifiers)):
                    raise ValueError('%s:%d repeats a track ID' % (
                        name, line_number
                    ))
                total_detections += count
            total_frames += len(lines)
    print(
        'VALID archive=%s sequences=%d frames=%d detections=%d'
        % (archive_path, len(expected), total_frames, total_detections)
    )


if __name__ == '__main__':
    main()
