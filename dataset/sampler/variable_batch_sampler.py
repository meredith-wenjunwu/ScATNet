from typing import Optional
import torch
from torch.utils.data.sampler import Sampler
from typing import Optional
import torch.distributed as dist
import math
from itertools import product
import argparse
import random
import numpy as np


def make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def bound_fn(min_val, max_val, value):
    return max(min_val, min(max_val, value))

class VariableBatchSampler(Sampler):
    def __init__(self, im_width: int, im_height: int, crop_width: int, crop_height: int,
                 n_data_samples: int, batch_size: int=1, is_training: Optional[bool] = False):
        self.image_sizes = [(im_height, im_width)]
        if is_training:
            self.image_sizes = []
            for sc in np.linspace(0.5, 1.0, 5):
                new_width = make_divisible(int(im_width * sc), crop_width)
                new_height = make_divisible(int(im_height * sc), crop_height)
                if (new_height, new_width) not in self.image_sizes:
                    self.image_sizes.append((new_height, new_width))

        n_gpus: int = max(1, torch.cuda.device_count())

        n_samples_per_gpu = int(math.ceil(n_data_samples * 1.0 / n_gpus))
        total_size = n_samples_per_gpu * n_gpus

        indexes = [idx for idx in range(n_data_samples)]
        # This ensures that we can divide the batches evenly across GPUs
        indexes += indexes[:(total_size - n_data_samples)]
        assert total_size == len(indexes)
        self.img_indices = indexes
        self.n_samples = total_size

        self.shuffle = True if is_training else False
        self.batch_size = batch_size

    def __len__(self):
        return self.n_samples//self.batch_size

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.img_indices)
            random.shuffle(self.image_sizes)

        start_index = 0
        while start_index < self.n_samples:
            im_h, im_w = random.choice(self.image_sizes)

            end_index = min(start_index + self.batch_size, self.n_samples)
            batch_ids = self.img_indices[start_index:end_index]
            start_index += self.batch_size

            if len(batch_ids) > 0:
                batch = [(im_h, im_w, b_id) for b_id in batch_ids]
                yield batch
