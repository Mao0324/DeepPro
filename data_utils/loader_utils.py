from pathlib import Path


def validate_frame_pairs(
    image_root,
    label_root,
    images,
    labels,
    sequence_name,
    minimum_frames=1,
):
    if not Path(image_root).is_dir():
        raise FileNotFoundError('Image directory does not exist: %s' % image_root)
    if not Path(label_root).is_dir():
        raise FileNotFoundError('Label directory does not exist: %s' % label_root)
    if len(images) != len(labels):
        raise ValueError(
            '%s has %d images but %d labels.'
            % (sequence_name, len(images), len(labels))
        )
    if len(images) < minimum_frames:
        raise ValueError(
            '%s has %d frames; at least %d are required.'
            % (sequence_name, len(images), minimum_frames)
        )

    image_stems = [Path(name).stem for name in images]
    label_stems = [Path(name).stem for name in labels]
    if image_stems != label_stems:
        mismatch_index = next(
            index
            for index, pair in enumerate(zip(image_stems, label_stems))
            if pair[0] != pair[1]
        )
        raise ValueError(
            '%s image/label names do not match at index %d: %s versus %s.'
            % (
                sequence_name,
                mismatch_index,
                images[mismatch_index],
                labels[mismatch_index],
            )
        )
