from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm


class Labels:
    safe = 0
    not_safe = 1


class NsfwDataset(Dataset):
    class Metadata:
        def __init__(self, root, label, shuffle=True):
            if label == Labels.safe:
                self.path = os.path.join(root, "safe")
            else:
                self.path = os.path.join(root, "not-safe-singleavg")
            self.file_list = glob.glob(os.path.join(self.path, "*"))
            self.size = len(self.file_list)

            if shuffle:
                random.shuffle(self.file_list)

            self.file_gen = iter(self.file_list)

    def __init__(self, root: str, transform=None, training=True):
        self.mean = None
        self.std = None
        self.root = root
        self.shuffle = True if training else False
        self.label_metadata = {
            Labels.safe: self.Metadata(self.root, Labels.safe, self.shuffle),
            Labels.not_safe: self.Metadata(self.root, Labels.not_safe, self.shuffle),
        }
        self.transform = transform

    def __getitem__(self, idx):
        target = random.choice([Labels.safe, Labels.not_safe])
        metadata = self.label_metadata[target]
        target_file = next(metadata.file_gen)
        img = Image.open(target_file).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, float(target), target_file

    def __len__(self):
        return self.label_metadata[Labels.safe].size + self.label_metadata[Labels.not_safe].size

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            whole_files = self.label_metadata[Labels.safe].file_list + self.label_metadata[Labels.not_safe].file_list
            for image_path in tqdm(whole_files[:3000]):
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255


# trsfm = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((120, 120))
#     # transforms.Normalize((0.1307,), (0.3081,))
# ])
#
# home = '/Users/jin/PycharmProjects/nsfw_torch'
# dataset = NsfwDataset(root=home + "/data/nsfw/", transform=trsfm)
# for i, data in enumerate(tqdm(dataset)):
#     print(data)
#     if i == 10:
#         break


# print(dataset.calc_statistics())
# print(len(dataset))
#

#
# dataset.mean
# Out[3]: array([0.54134208, 0.51423307, 0.49567292])
# dataset.std
# Out[4]: array([0.61925002, 0.59373791, 0.57680947])

