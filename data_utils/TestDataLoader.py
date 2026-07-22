import os
import torch
import numpy as np
from PIL import Image
import math
import random
from torch.utils.data import Dataset
from data_utils.loader_utils import (
    SATVIDEO_V1_DATASET,
    SATVIDEO_V1_TRAIN_MEAN,
    SATVIDEO_V1_TRAIN_STD,
    discover_split_sequences,
    validate_frame_pairs,
)


class TestSeqDataLoader(Dataset):
    def __init__(
        self,
        dataset,
        data_root,
        samplelist,
        seq_len=100,
        transform=None,
        load_annotations=True,
    ):
        self.dataset = dataset
        self.data_root = data_root
        self.samplelist = samplelist
        self.seq_len = seq_len
        self.transform = transform
        self.load_annotations = load_annotations
        if 'NUDT-MIRSDT' in dataset:
            self.train_mean = 105.4025
            self.train_std = 26.6452
        elif dataset == 'IRDST-simulation':
            self.train_mean = 106.8523
            self.train_std = 56.9243
        elif dataset == 'RGB-T':
            self.train_mean = 85.0799
            self.train_std = 47.4845
        elif dataset == 'SatVideoIRSDT':
            self.train_mean = 111.47
            self.train_std = 22.43
        elif dataset == SATVIDEO_V1_DATASET:
            # Always use training-split statistics for validation/test data.
            self.train_mean = SATVIDEO_V1_TRAIN_MEAN
            self.train_std = SATVIDEO_V1_TRAIN_STD
        elif dataset == 'IRSatVideo-LEO':
            self.train_mean = 72.104
            self.train_std = 12.303

    def __len__(self):
        return len(self.samplelist)

    def get_image_label(self, image_path, label_path, centroid_path=None):
        with Image.open(image_path) as image_file:
            if self.dataset == 'IRSatVideo-LEO':
                image_file = image_file.resize([512, 512])
            image = np.array(image_file, dtype=np.float32)
        if image.ndim == 3:
            image = image[:,:,0]
        image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)

        if not self.load_annotations:
            return image, None, None

        with Image.open(label_path) as label_file:
            label = np.array(label_file, dtype=np.float32) / 255.
        if label.ndim == 3:
            label = label[:,:,0]
        label[label > 0] = 1.
        label = np.expand_dims(label, axis=0)

        if 'NUDT-MIRSDT' in self.dataset:
            with Image.open(centroid_path) as centroid_file:
                centroid = np.array(centroid_file, dtype=np.float32) / 255.
            centroid = np.expand_dims(centroid, axis=0)
        else:
            centroid = label

        return image, label, centroid

    def sample_sequence(self, idx):
        sample = self.samplelist[idx]   ## frame：各帧在序列中的顺序（0开始）
        if len(sample) == 0:
            raise ValueError('Validation sample must contain at least one frame.')
        first_frame = sample[0][2]
        end_frame = sample[-1][2]

        image_path, label_path, _, centroid_path = sample[0]
        image, label, centroid = self.get_image_label(
            image_path,
            label_path,
            centroid_path,
        )
        _, _, h, w = image.shape
        images = np.empty(
            [1, len(sample), h, w],
            dtype=image.dtype,
        )
        images[:, 0:1, :, :] = image
        if self.load_annotations:
            labels = np.empty(
                [len(sample), h, w],
                dtype=label.dtype,
            )
            centroids = np.empty(
                [len(sample), h, w],
                dtype=centroid.dtype,
            )
            labels[0:1, :, :] = label
            centroids[0:1, :, :] = centroid

        for i in range(1, len(sample)):
            image_path, label_path, _, centroid_path = sample[i]
            image, label, centroid = self.get_image_label(
                image_path,
                label_path,
                centroid_path,
            )
            images[:, i:i+1, :, :] = image
            if self.load_annotations:
                labels[i:i+1, :, :] = label
                centroids[i:i+1, :, :] = centroid

        images = (images - self.train_mean) / self.train_std
        t = len(sample)
        if t < self.seq_len:
            padding = self.seq_len - t
            images = np.concatenate((images, np.zeros(
                [1, padding, h, w], dtype=images.dtype
            )), axis=1)
            if self.load_annotations:
                labels = np.concatenate((labels, np.zeros(
                    [padding, h, w], dtype=labels.dtype
                )), axis=0)
                centroids = np.concatenate((centroids, np.zeros(
                    [padding, h, w], dtype=centroids.dtype
                )), axis=0)

        # if self.transform is not None:
        #     sample = self.transform(sample)   #########################

        images = torch.from_numpy(images)
        if self.load_annotations:
            labels = torch.from_numpy(labels)
            centroids = torch.from_numpy(centroids)
        else:
            labels = torch.empty(0, dtype=images.dtype)
            centroids = torch.empty(0, dtype=images.dtype)
        # labels = 0
        # centroids = 0

        return images, labels, centroids, [first_frame, end_frame]

    def __getitem__(self, idx):
        images, labels, centroids, first_end = self.sample_sequence(idx)

        return images, labels, centroids, first_end



class TestIRSeqDataLoader(object):
    def __init__(
        self,
        dataset='NUDT-MIRSDT',
        data_root='./datasets/IRSeq',
        seq_len=100,
        cat_len=10,
        transform=None,
        load_annotations=True,
    ):
        if seq_len <= 0:
            raise ValueError('seq_len must be positive.')
        if cat_len < 0 or cat_len >= seq_len:
            raise ValueError('cat_len must satisfy 0 <= cat_len < seq_len.')
        self.dataset = dataset
        self.data_root = data_root
        self.seq_len = seq_len
        self.cat_len = cat_len
        self.transform = transform
        self.load_annotations = load_annotations
        if dataset == SATVIDEO_V1_DATASET:
            self.seq_list_file = None
            self.seq_names = discover_split_sequences(data_root, 'val')
        elif 'NUDT-MIRSDT' in dataset or 'RGB-T' in dataset:
            self.seq_list_file = os.path.join(data_root, 'test.txt')
        elif dataset == 'IRDST-simulation':
            self.seq_list_file = os.path.join(data_root, 'img_idx/test_IRDST-simulation.txt')
        elif dataset == 'SatVideoIRSDT':
            self.seq_list_file = os.path.join(data_root, 'val.txt')
        elif dataset == 'IRSatVideo-LEO':
            self.seq_list_file = os.path.join(data_root, 'annotations/val_sequences.txt')
        else:
            raise ValueError('Unsupported test dataset: %s' % dataset)
        if self.seq_list_file is not None:
            self._check_preprocess()
            self.seq_names = list(dict.fromkeys([
                x.split('/')[0] for x in self.ann_f
            ]))
        # self.seq_names = list([str(self.ann_f)])

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]

        if 'NUDT-MIRSDT' in self.dataset:
            image_root = os.path.join(self.data_root, seq_name, 'images')
            label_root = os.path.join(self.data_root, seq_name, 'masks').replace('NUDT-MIRSDT-Noise/'+self.dataset, 'NUDT-MIRSDT')
            images = np.sort(os.listdir(image_root))
            labels = np.sort(os.listdir(label_root))
            centroid_root = os.path.join(
                self.data_root,
                seq_name,
                'masks_centroid',
            )
            if not os.path.isdir(centroid_root):
                centroid_root = os.path.join(
                    os.path.dirname(self.data_root),
                    'NUDT-MIRSDT',
                    seq_name,
                    'masks_centroid',
                )
            if not os.path.isdir(centroid_root):
                raise FileNotFoundError(
                    'Centroid directory does not exist for %s.' % seq_name
                )
            centroid_files = np.sort(os.listdir(centroid_root))
        elif 'RGB-T' in self.dataset:
            image_root = os.path.join(self.data_root, 'test2017', seq_name, '01')
            label_root = os.path.join(self.data_root, 'segmentations', seq_name)
            images = np.sort(os.listdir(image_root))
            labels = np.sort(os.listdir(label_root))
        elif self.dataset in ['IRDST-simulation', 'IRSatVideo-LEO']:
            image_root = os.path.join(self.data_root, 'images', seq_name)
            label_root = os.path.join(self.data_root, 'masks', seq_name)
            images = os.listdir(image_root)
            labels = os.listdir(label_root)
            images.sort(key=lambda x:int(x.split('.')[0]))
            labels.sort(key=lambda x:int(x.split('.')[0]))
        elif self.dataset in ['SatVideoIRSDT', SATVIDEO_V1_DATASET]:
            image_root = os.path.join(self.data_root, 'val', seq_name, 'img')
            label_root = os.path.join(self.data_root, 'val', seq_name, 'mask')
            images = np.sort(os.listdir(image_root))
            labels = np.sort(os.listdir(label_root))

        validate_frame_pairs(
            image_root,
            label_root,
            images,
            labels,
            seq_name,
        )
        if 'NUDT-MIRSDT' in self.dataset:
            validate_frame_pairs(
                label_root,
                centroid_root,
                labels,
                centroid_files,
                '%s centroid masks' % seq_name,
            )

        samplelist = []
        num_sample = max(
            1,
            math.ceil(
                (len(images) - self.cat_len)
                / (self.seq_len - self.cat_len)
            ),
        )
        for i in range(num_sample):
            last_frame = min(len(images), (i+1)*(self.seq_len-self.cat_len)+self.cat_len)
            sample = [(
                os.path.join(image_root, images[x]),
                os.path.join(label_root, labels[x]),
                x,
                os.path.join(centroid_root, centroid_files[x])
                if 'NUDT-MIRSDT' in self.dataset else None,
            )
                      for x in range(max(0, last_frame-self.seq_len), last_frame)]
            samplelist.extend([sample])

        seq_dataset = TestSeqDataLoader(
            self.dataset,
            self.data_root,
            samplelist,
            self.seq_len,
            self.transform,
            load_annotations=self.load_annotations,
        )

        return seq_dataset

    def flatten_windows(self):
        """Build one deterministic window dataset for persistent validation workers."""
        samplelist = []
        for seq_idx in range(len(self)):
            samplelist.extend(self[seq_idx].samplelist)
        return TestSeqDataLoader(
            self.dataset,
            self.data_root,
            samplelist,
            self.seq_len,
            self.transform,
            load_annotations=self.load_annotations,
        )

    def _check_preprocess(self):
        if not os.path.isfile(self.seq_list_file):
            raise FileNotFoundError('No such file: %s.' % self.seq_list_file)
        self.ann_f = np.atleast_1d(
            np.loadtxt(self.seq_list_file, dtype=bytes).astype(str)
        )
        return True
