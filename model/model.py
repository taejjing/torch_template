import torch.nn as nn
import torch
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import resnet50


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class NsfwResnet(BaseModel):
    def __init__(self, freeze=False):
        super().__init__()
        self.net = resnet50()
        state_dict = torch.load('./pretrained_model/resnet50-19c8e357.pth')
        self.net.load_state_dict(state_dict)

        self.net.fc = nn.Sequential(
            nn.Linear(self.net.fc.in_features, 512),
            nn.ReLU(True),
            nn.Linear(512, 1)
        )
        if freeze:
            self._freeze_net()


    def forward(self, x):
        x = self.net(x)
        x = torch.reshape(x, (-1,))
        return x

    def _freeze_net(self):
        print("Feature Extraction, model freeze")
        for param in self.net.parameters():
            param.requires_grad = False

        for param in self.net.fc.parameters():
            param.requires_grad = True
