#!/usr/bin/env python3
"""Generate challenge TXT files with persistent target trajectory IDs.

The input directory may be either:

1. a directory containing detection-only ``*.txt`` files produced by
   ``test.py --centroid_txt``; or
2. a directory containing one PNG sequence, or one PNG subdirectory per
   sequence.

The challenge field order is always ``target_id x y``.  Existing outputs are
never overwritten unless ``--overwrite`` is supplied explicitly.
"""

import argparse
import math
import os
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise ImportError(
        "seg2tracked_centroid_txt.py requires scipy for global track assignment."
    ) from exc


@dataclass
class Detection:
    x: float
    y: float
    area: Optional[float] = None


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    area: Optional[float]
    last_frame: int
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1

    def predict(self, frame_idx: int) -> Tuple[float, float]:
        delta = frame_idx - self.last_frame
        return self.x + self.vx * delta, self.y + self.vy * delta


class CentroidTracker:
    """Constant-velocity tracker with gated global bipartite assignment."""

    _INVALID_COST = 1.0e12

    def __init__(
        self,
        max_distance: float,
        distance_growth: float,
        max_missed: int,
        area_weight: float,
        max_area_ratio: float,
        velocity_smoothing: float,
    ) -> None:
        self.max_distance = max_distance
        self.distance_growth = distance_growth
        self.max_missed = max_missed
        self.area_weight = area_weight
        self.max_area_ratio = max_area_ratio
        self.velocity_smoothing = velocity_smoothing
        self._tracks: Dict[int, Track] = {}
        self._next_track_id = 1

    def _active_tracks(self, frame_idx: int) -> List[Track]:
        active = []
        expired = []
        for track_id, track in self._tracks.items():
            missing_frames = frame_idx - track.last_frame - 1
            if missing_frames <= self.max_missed:
                active.append(track)
            else:
                expired.append(track_id)
        for track_id in expired:
            del self._tracks[track_id]
        return sorted(active, key=lambda item: item.track_id)

    def _pair_cost(
        self, track: Track, detection: Detection, frame_idx: int
    ) -> float:
        predicted_x, predicted_y = track.predict(frame_idx)
        distance = math.hypot(
            detection.x - predicted_x, detection.y - predicted_y
        )
        missing_frames = max(0, frame_idx - track.last_frame - 1)
        distance_limit = (
            self.max_distance + self.distance_growth * missing_frames
        )
        if distance > distance_limit:
            return self._INVALID_COST

        cost = distance
        if (
            track.area is not None
            and detection.area is not None
            and track.area > 0
            and detection.area > 0
        ):
            area_ratio = max(
                track.area / detection.area, detection.area / track.area
            )
            if self.max_area_ratio > 0 and area_ratio > self.max_area_ratio:
                return self._INVALID_COST
            cost += self.area_weight * abs(
                math.log(detection.area / track.area)
            )
        return cost

    def _new_track(self, detection: Detection, frame_idx: int) -> Track:
        track = Track(
            track_id=self._next_track_id,
            x=detection.x,
            y=detection.y,
            area=detection.area,
            last_frame=frame_idx,
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _update_track(
        self, track: Track, detection: Detection, frame_idx: int
    ) -> None:
        delta = frame_idx - track.last_frame
        measured_vx = (detection.x - track.x) / delta
        measured_vy = (detection.y - track.y) / delta
        if track.hits == 1:
            track.vx = measured_vx
            track.vy = measured_vy
        else:
            old_weight = self.velocity_smoothing
            new_weight = 1.0 - old_weight
            track.vx = old_weight * track.vx + new_weight * measured_vx
            track.vy = old_weight * track.vy + new_weight * measured_vy
        track.x = detection.x
        track.y = detection.y
        track.area = detection.area
        track.last_frame = frame_idx
        track.hits += 1

    def update(
        self, frame_idx: int, detections: Sequence[Detection]
    ) -> List[Tuple[int, Detection]]:
        detections = sorted(detections, key=lambda item: (item.x, item.y))
        tracks = self._active_tracks(frame_idx)
        assignments: List[Tuple[int, Detection]] = []
        matched_detection_indices = set()

        if tracks and detections:
            cost_matrix = np.empty(
                (len(tracks), len(detections)), dtype=np.float64
            )
            for row, track in enumerate(tracks):
                for column, detection in enumerate(detections):
                    cost_matrix[row, column] = self._pair_cost(
                        track, detection, frame_idx
                    )

            valid_costs = cost_matrix[cost_matrix < self._INVALID_COST]
            unmatched_cost = (
                float(valid_costs.max()) + 1.0 if valid_costs.size else 1.0
            )
            # A dummy column for every track turns the full Hungarian
            # assignment into a partial assignment.  Without these columns,
            # an invalid pair can displace a cheaper valid pair when several
            # tracks compete for one gated detection.
            assignment_costs = np.full(
                (len(tracks), len(detections) + len(tracks)),
                unmatched_cost,
                dtype=np.float64,
            )
            assignment_costs[:, : len(detections)] = cost_matrix
            track_indices, detection_indices = linear_sum_assignment(
                assignment_costs
            )
            for track_index, detection_index in zip(
                track_indices.tolist(), detection_indices.tolist()
            ):
                if detection_index >= len(detections):
                    continue
                if (
                    cost_matrix[track_index, detection_index]
                    >= self._INVALID_COST
                ):
                    continue
                track = tracks[track_index]
                detection = detections[detection_index]
                self._update_track(track, detection, frame_idx)
                assignments.append((track.track_id, detection))
                matched_detection_indices.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue
            track = self._new_track(detection, frame_idx)
            assignments.append((track.track_id, detection))

        return sorted(assignments, key=lambda item: item[0])


def _normalise_mask(
    image: np.ndarray, path: Path, integer_scale: str
) -> np.ndarray:
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(
                "Unsupported channel count in mask %s: %s"
                % (path, image.shape)
            )
    if image.ndim != 2:
        raise ValueError("Mask must be two-dimensional: %s" % path)

    values = image.astype(np.float32, copy=False)
    minimum = float(values.min()) if values.size else 0.0
    maximum = float(values.max()) if values.size else 0.0
    if minimum < 0:
        raise ValueError("Mask contains negative values: %s" % path)
    if np.issubdtype(image.dtype, np.integer) and integer_scale == "dtype":
        scale = float(np.iinfo(image.dtype).max)
    elif maximum <= 1.0:
        scale = 1.0
    elif maximum <= 255.0:
        scale = 255.0
    elif np.issubdtype(image.dtype, np.integer):
        scale = float(np.iinfo(image.dtype).max)
    else:
        raise ValueError(
            "Floating-point mask values must be in [0, 1] or [0, 255]: %s"
            % path
        )
    return values / scale


def _extract_detections(
    mask_path: Path, threshold: float, min_area: int, integer_scale: str
) -> List[Detection]:
    image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Cannot read mask: %s" % mask_path)
    probability = _normalise_mask(image, mask_path, integer_scale)
    binary_mask = np.uint8(probability > threshold)
    number, labels, stats, component_centroids = (
        cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    )

    detections = []
    for label in range(1, number):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        local_labels = labels[top : top + height, left : left + width]
        local_mask = np.uint8(local_labels == label) * 255
        contours, _ = cv2.findContours(
            local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(contour)
        else:
            moments = {"m00": 0.0}
        if moments["m00"] != 0:
            x_coordinate = left + moments["m10"] / moments["m00"]
            y_coordinate = top + moments["m01"] / moments["m00"]
        else:
            x_coordinate, y_coordinate = component_centroids[label]
        detections.append(
            Detection(
                x=round(float(x_coordinate), 2),
                y=round(float(y_coordinate), 2),
                area=float(area),
            )
        )
    return detections


def _format_line(
    frame_idx: int, assignments: Sequence[Tuple[int, Detection]]
) -> str:
    fields = ["%05d" % frame_idx, str(len(assignments))]
    for track_id, detection in sorted(assignments, key=lambda item: item[0]):
        fields.extend(
            [
                str(track_id),
                "%.2f" % detection.x,
                "%.2f" % detection.y,
            ]
        )
    return " ".join(fields)


def _validate_frame_indices(frame_indices: Sequence[int], source: Path) -> None:
    if not frame_indices:
        raise ValueError("Sequence contains no frames: %s" % source)
    if any(current <= previous for previous, current in zip(
        frame_indices, frame_indices[1:]
    )):
        raise ValueError("Frame numbers are not strictly increasing: %s" % source)
    for previous, current in zip(frame_indices, frame_indices[1:]):
        if current != previous + 1:
            raise ValueError(
                "Missing frame between %05d and %05d in %s"
                % (previous, current, source)
            )


def _parse_detection_txt(path: Path) -> List[Tuple[int, List[Detection]]]:
    frames = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            stripped = raw_line.strip()
            if not stripped:
                raise ValueError(
                    "Blank line at %s:%d" % (path, line_number)
                )
            fields = stripped.split()
            if len(fields) < 2:
                raise ValueError(
                    "Incomplete line at %s:%d" % (path, line_number)
                )
            try:
                frame_idx = int(fields[0])
                target_count = int(fields[1])
            except ValueError as exc:
                raise ValueError(
                    "Invalid frame/count at %s:%d" % (path, line_number)
                ) from exc
            expected_fields = 2 + target_count * 3
            if target_count < 0 or len(fields) != expected_fields:
                raise ValueError(
                    "Expected %d fields but found %d at %s:%d"
                    % (expected_fields, len(fields), path, line_number)
                )
            detections = []
            for offset in range(2, len(fields), 3):
                try:
                    # The source target ID is intentionally ignored: this script
                    # assigns persistent trajectory IDs from the coordinates.
                    int(fields[offset])
                    x_coordinate = float(fields[offset + 1])
                    y_coordinate = float(fields[offset + 2])
                except ValueError as exc:
                    raise ValueError(
                        "Invalid target at %s:%d" % (path, line_number)
                    ) from exc
                if not (
                    math.isfinite(x_coordinate) and math.isfinite(y_coordinate)
                ):
                    raise ValueError(
                        "Non-finite coordinate at %s:%d" % (path, line_number)
                    )
                detections.append(Detection(x_coordinate, y_coordinate))
            frames.append((frame_idx, detections))
    _validate_frame_indices([frame[0] for frame in frames], path)
    return frames


def _numeric_png_frames(
    sequence_dir: Path,
) -> List[Tuple[int, Path]]:
    png_files = sorted(
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    frames = []
    seen = set()
    for path in png_files:
        try:
            frame_idx = int(path.stem)
        except ValueError as exc:
            raise ValueError(
                "PNG filename must be a numeric frame number: %s" % path
            ) from exc
        if frame_idx in seen:
            raise ValueError(
                "Duplicate numeric frame number %d in %s"
                % (frame_idx, sequence_dir)
            )
        seen.add(frame_idx)
        frames.append((frame_idx, path))
    frames.sort(key=lambda item: item[0])
    _validate_frame_indices([frame[0] for frame in frames], sequence_dir)
    return frames


def _discover_input(
    input_root: Path, sequence_name: Optional[str]
) -> Tuple[str, List[Tuple[str, object]]]:
    if not input_root.is_dir():
        raise ValueError("Input root is not a directory: %s" % input_root)

    txt_files = sorted(
        path
        for path in input_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    )
    direct_pngs = [
        path
        for path in input_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    ]
    png_subdirs = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir()
        and any(
            child.is_file() and child.suffix.lower() == ".png"
            for child in path.iterdir()
        )
    )

    modes = sum(bool(items) for items in (txt_files, direct_pngs, png_subdirs))
    if modes == 0:
        raise ValueError(
            "No TXT files or PNG sequence directories found in %s"
            % input_root
        )
    if modes > 1:
        raise ValueError(
            "Input mixes TXT files, direct PNGs, or PNG subdirectories: %s"
            % input_root
        )

    if txt_files:
        if sequence_name:
            raise ValueError("--sequence-name is only valid for direct PNG input")
        sequences = [(path.stem, path) for path in txt_files]
        for name, _ in sequences:
            _validate_sequence_name(name)
        return "txt", sequences
    if direct_pngs:
        name = sequence_name or input_root.name
        _validate_sequence_name(name)
        return "mask", [(name, input_root)]
    if sequence_name:
        raise ValueError("--sequence-name is only valid for direct PNG input")
    sequences = [(path.name, path) for path in png_subdirs]
    for name, _ in sequences:
        _validate_sequence_name(name)
    return "mask", sequences


def _validate_sequence_name(name: str) -> None:
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError("Unsafe sequence name: %r" % name)


def _track_frames(
    frames: Iterable[Tuple[int, List[Detection]]], args: argparse.Namespace
) -> Tuple[List[str], int, int]:
    tracker = CentroidTracker(
        max_distance=args.max_distance,
        distance_growth=args.distance_growth,
        max_missed=args.max_missed,
        area_weight=args.area_weight,
        max_area_ratio=args.max_area_ratio,
        velocity_smoothing=args.velocity_smoothing,
    )
    tracked_frames = []
    track_observations = Counter()
    for frame_idx, detections in frames:
        assignments = tracker.update(frame_idx, detections)
        tracked_frames.append((frame_idx, assignments))
        track_observations.update(track_id for track_id, _ in assignments)

    retained_ids = sorted(
        track_id for track_id, observations in track_observations.items()
        if observations >= args.min_track_observations
    )
    identifier_map = {
        track_id: index for index, track_id in enumerate(retained_ids, start=1)
    }
    lines = []
    target_total = 0
    for frame_idx, assignments in tracked_frames:
        retained = [
            (identifier_map[track_id], detection)
            for track_id, detection in assignments
            if track_id in identifier_map
        ]
        lines.append(_format_line(frame_idx, retained))
        target_total += len(retained)
    return lines, target_total, len(retained_ids)


def _atomic_write_text(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write("\n".join(lines))
            output.write("\n")
        os.replace(temporary_name, str(path))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_zip(zip_path: Path, txt_paths: Sequence[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % zip_path.name,
        suffix=".tmp",
        dir=str(zip_path.parent),
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary_name, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for txt_path in txt_paths:
                # Competition packaging requires TXT files at the ZIP root.
                archive.write(str(txt_path), arcname=txt_path.name)
        os.replace(temporary_name, str(zip_path))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError(
            "value must be finite and greater than zero"
        )
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError(
            "value must be finite and non-negative"
        )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assign persistent trajectory IDs and generate SatVideoIRSTD "
            "competition TXT files."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "Directory containing detection TXT files, one PNG sequence, "
            "or one PNG subdirectory per sequence"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for tracked TXT files",
    )
    parser.add_argument(
        "--sequence-name",
        help="Output sequence name when --input-root directly contains PNGs",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Mask probability threshold in [0, 1] (default: 0.5)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1,
        help="Minimum connected-component pixel area in mask mode (default: 1)",
    )
    parser.add_argument(
        "--integer-scale",
        choices=("auto", "dtype"),
        default="auto",
        help=(
            "Integer PNG normalization: auto preserves 0/1 binary masks; "
            "dtype always divides by the dtype maximum for probability maps"
        ),
    )
    parser.add_argument(
        "--max-distance",
        type=_positive_float,
        default=20.0,
        help="Maximum predicted centroid matching distance in pixels (default: 20)",
    )
    parser.add_argument(
        "--distance-growth",
        type=_nonnegative_float,
        default=5.0,
        help="Additional distance gate per missing frame (default: 5)",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=2,
        help="Missing frames for which a track remains active (default: 2)",
    )
    parser.add_argument(
        "--min-track-observations",
        type=int,
        default=1,
        help=(
            "Discard trajectories observed in fewer frames than this value "
            "after tracking (default: 1)"
        ),
    )
    parser.add_argument(
        "--area-weight",
        type=_nonnegative_float,
        default=2.0,
        help="Area-change penalty in mask mode (default: 2)",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=_nonnegative_float,
        default=4.0,
        help="Maximum matched area ratio; 0 disables the gate (default: 4)",
    )
    parser.add_argument(
        "--velocity-smoothing",
        type=float,
        default=0.5,
        help="Previous-velocity weight in [0, 1] (default: 0.5)",
    )
    parser.add_argument(
        "--zip-output",
        type=Path,
        help="Optionally create a competition-ready flat ZIP archive",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of generated TXT/ZIP paths",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.min_area < 1:
        parser.error("--min-area must be at least 1")
    if args.max_missed < 0:
        parser.error("--max-missed must be non-negative")
    if args.min_track_observations < 1:
        parser.error("--min-track-observations must be at least 1")
    if (
        not math.isfinite(args.velocity_smoothing)
        or not 0.0 <= args.velocity_smoothing <= 1.0
    ):
        parser.error("--velocity-smoothing must be between 0 and 1")

    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if input_root == output_dir:
        parser.error("--output-dir must differ from --input-root")
    if args.zip_output:
        args.zip_output = args.zip_output.expanduser().resolve()
        if args.zip_output.suffix.lower() != ".zip":
            parser.error("--zip-output must end with .zip")

    input_mode, sequence_inputs = _discover_input(
        input_root, args.sequence_name
    )
    output_paths = [
        output_dir / (sequence_name + ".txt")
        for sequence_name, _ in sequence_inputs
    ]
    if args.zip_output and args.zip_output in output_paths:
        parser.error("--zip-output must differ from every generated TXT path")
    collisions = [path for path in output_paths if path.exists()]
    if args.zip_output and args.zip_output.exists():
        collisions.append(args.zip_output)
    if collisions and not args.overwrite:
        parser.error(
            "Refusing to overwrite existing output(s): %s"
            % ", ".join(str(path) for path in collisions)
        )

    generated: Dict[str, Tuple[List[str], int, int]] = {}
    for sequence_name, source in sequence_inputs:
        if input_mode == "txt":
            frames = _parse_detection_txt(source)
        else:
            mask_frames = _numeric_png_frames(source)
            frames = [
                (
                    frame_idx,
                    _extract_detections(
                        mask_path,
                        args.threshold,
                        args.min_area,
                        args.integer_scale,
                    ),
                )
                for frame_idx, mask_path in mask_frames
            ]
        generated[sequence_name] = _track_frames(frames, args)

    output_dir.mkdir(parents=True, exist_ok=True)
    for output_path, (sequence_name, _) in zip(output_paths, sequence_inputs):
        lines, target_total, track_total = generated[sequence_name]
        _atomic_write_text(output_path, lines)
        print(
            "%s: %d frames, %d detections, %d trajectories -> %s"
            % (
                sequence_name,
                len(lines),
                target_total,
                track_total,
                output_path,
            )
        )

    if args.zip_output:
        _atomic_write_zip(args.zip_output, output_paths)
        print("Competition ZIP (flat TXT layout): %s" % args.zip_output)


if __name__ == "__main__":
    main()
