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
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            # transforms.ColorJitter(brightness=0.5, saturation=0.5, hue=0.5),
            transforms.RandomRotation([-8, +8]),
            transforms.Normalize((0.5413, 0.5142, 0.4956), (0.6192, 0.5937, 0.5768)),
        ])
        self.data_dir = data_dir
        self.dataset = NsfwDataset(root=data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# home = '/Users/jin/PycharmProjects/nsfw_torch'
# dataloader = NsfwDataLoader(data_dir=home+"/data/nsfw/", batch_size=32, num_workers=4)
#
# for i, (data, target) in enumerate(tqdm(dataloader)):
#     pass



