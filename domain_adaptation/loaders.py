""" Some data loading utilities """
from bisect import bisect
from os import listdir, walk
from os.path import join, isdir, isfile
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

import pytorch_lightning as pl


# Utils to handle newer PyTorch Lightning changes from version 0.6
# ==================================================================================================== ##


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper

# class ObservationsDataset(Dataset):
#     """Simple duckietown image observations datasets"""
#
#     # Init the data
#     def __init__(self, root, transform, train_split=0.8,  train=True):
#         self._transform = transform
#         length = len(listdir(root))
#         self._files = [
#             join(root, sd)
#             for sd in listdir(root) if isdir(join(root, sd))
#             ]
#
#         index = int((length * train_split) // 1)
#
#         if train:
#             self._files = self._files[:index]
#             self._len = index
#         else:
#             self._files = self._files[index:]
#             self._len = length - index
#
#     def __getitem__(self, idx):
#         print("idx", idx)
#         return self._transform(self._files[idx])
#
#     def __len__(self):
#         return self._len

class ObservationsDataset(Dataset):
    """Simple duckietown image observations datasets"""

    # Init the data
    def __init__(self, root, transform, crop=False):
        self._len = len(listdir(root))
        self._transform = transform
        #self._files = root
        self._files = [join(root, f) for f in listdir(root) if isfile(join(root, f))]
        self._crop = crop
        self._filenames = []
        for (dirpath, dirnames, filenames) in walk(root):
            for f in filenames:
                if f.endswith('.npy'):

                    self._filenames.append(f.rstrip('.npy'))

    def __getitem__(self, idx):

        img = np.load(self._files[idx])
        filename = self._filenames[idx]
        if self._crop:
            img = img[int(img.shape[0]/3):, :, :]
        transformed = self._transform(img)
        return transformed, filename

    def __len__(self):
        return self._len


class LoadDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def get_target_from_path(self, path):
        # Implement your target extraction here
        return torch.tensor([0])

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index])
        #y = self.get_target_from_path(self.image_paths[index])
        if self.transform:
            x = self.transform(x)

        return x  # , y

    def __len__(self):
        return len(self.image_paths)




class _RolloutDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    def __init__(self, root, transform, buffer_size=100, train=True): # pylint: disable=too-many-arguments
        self._transform = transform

        self._files = [
            join(root, sd, ssd)
            for sd in listdir(root) if isdir(join(root, sd))
            for ssd in listdir(join(root, sd))]

        if train:
            #self._files = self._files[:-600]
            self._files = self._files[:-500]
        else:
            #self._files = self._files[-600:]
            self._files = self._files[-500:]

        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []
        self._cum_size = [0]

        # progress bar
        pbar = tqdm(total=len(self._buffer_fnames),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with np.load(f) as data:
                self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                self._cum_size += [self._cum_size[-1] +
                                   self._data_per_sequence(data['rewards'].shape[0])]
            pbar.update(1)
        pbar.close()

    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len obs and seq_len next_obs sequences
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]

    def __getitem__(self, i):
        # binary search through cum_size
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        pass

    def _data_per_sequence(self, data_length):
        pass



class RolloutObservationDataset(_RolloutDataset): # pylint: disable=too-few-public-methods
    """ Encapsulates rollouts.
    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean
     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.
    Data are then provided in the form of images
    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data, else test
    """
    def _data_per_sequence(self, data_length):
        return data_length

    def _get_data(self, data, seq_index):
        return self._transform(data['observations'][seq_index])
