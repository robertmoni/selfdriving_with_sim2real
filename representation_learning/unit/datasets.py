import glob
import random
import os

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/sim" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/real" % mode) + "/*.*"))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        # z = np.load(self.files_A[index % len(self.files_A)])
        # y = np.load(self.files_B[index % len(self.files_B)])
        # item_A = self.transform(Image.fromarray(np.uint8(np.load(self.files_A[index % len(self.files_A)], allow_pickle=True))))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
            # item_B = self.transform(Image.fromarray(np.uint8(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)], allow_pickle=True))))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
            # item_B = self.transform(Image.fromarray(np.uint8(np.load(self.files_B[index % len(self.files_B)], allow_pickle=True))))
        return {"sim": item_A, "real": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
