from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from enum import Enum


class Labels(Enum):
    safe = 0
    unsafe = 1
    TEST = -1


class NsfwDataset(Dataset):
    class Metadata:
        def __init__(self, root: str, label: Labels):
            self.label = label

            if self.label == Labels.safe:
                self.path = os.path.join(root, "safe")
            elif self.label == Labels.unsafe:
                self.path = os.path.join(root, "unsafe")
            else:
                self.path = root

            self.file_list = glob.glob(os.path.join(self.path, "*"))
            self.size = len(self.file_list)
            self.files_with_label = list(zip(self.file_list, [self.label.value] * self.size)) # modified

    def __init__(self, root: str, transform=None, training=True, eval=False, calc_stats=False):
        self.mean = None
        self.std = None
        self.root = root
        self.training = training
        self.transform = transform

        if training or eval:
            self.label_metadata = {
                Labels.safe: self.Metadata(self.root, Labels.safe),
                Labels.unsafe: self.Metadata(self.root, Labels.unsafe)
            }
        else:
            self.label_metadata = {
                Labels.TEST: self.Metadata(self.root, Labels.TEST)
            }

        self.item_list = []
        for m in self.label_metadata.values():
            self.item_list.extend(m.files_with_label)
        
        if calc_stats:
            self.calc_statistics()
            print("Statistics from given Dataset")
            print(f"Mean: {self.mean} \nStd: {self.std}")


    def __getitem__(self, idx):
        item = self.item_list[idx]
        target_file = item[0]
        label = item[1]
        img = Image.open(target_file).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, float(label), target_file

    def __len__(self):
        return sum(map(lambda x: x.size, self.label_metadata.values()))

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            whole_files = list(map(lambda x: x[0], self.item_list))
            random.shuffle(whole_files)
            for image_path in tqdm(whole_files[:3000]):
                image = np.array(Image.open(image_path).convert("RGB")).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255
