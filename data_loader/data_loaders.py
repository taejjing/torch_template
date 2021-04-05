from torchvision import datasets, transforms
from .dataset import NsfwDataset
from base import BaseDataLoader
from tqdm import tqdm


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class NsfwDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, eval=False, calc_stats=False):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.RandomRotation([-8, +8]),
            transforms.Normalize((0.5413, 0.5142, 0.4956), (0.6192, 0.5937, 0.5768))
        ])
        self.data_dir = data_dir
        self.dataset = NsfwDataset(root=data_dir, transform=trsfm, training=training, eval=eval, calc_stats=calc_stats)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
