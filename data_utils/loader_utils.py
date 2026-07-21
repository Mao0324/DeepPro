from pathlib import Path


SATVIDEO_V1_DATASET = 'SatVideoIRSDT_v1'
SATVIDEO_V1_TRAIN_MEAN = 82.20451526467026
SATVIDEO_V1_TRAIN_STD = 50.753589902516666


def discover_split_sequences(data_root, split):
    """Discover ``<split>/<sequence>/{img,mask}`` sequence directories."""
    split_root = Path(data_root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(
            'Dataset split directory does not exist: %s' % split_root
        )

    sequence_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
    if not sequence_dirs:
        raise ValueError('No sequence directories found in %s.' % split_root)

    invalid = [
        path.name
        for path in sequence_dirs
        if not (path / 'img').is_dir() or not (path / 'mask').is_dir()
    ]
    if invalid:
        raise ValueError(
            'Sequences in %s must contain img and mask directories: %s'
            % (split_root, ', '.join(invalid[:10]))
        )
    return [path.name for path in sequence_dirs]


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
