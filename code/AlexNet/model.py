import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Alexnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernerl_size =11, stride = 4, padding=0)
        # self.pool1 = nn.MaxPool2d(kernel_size=3, stride = 2)
        # self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        # self.conv2 = nn.Conv2d(in_channels=63, out_channels=256, kernel_size=5, stride=1, padding=2)
        # self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # self.norm2 = nn.LocalResponseNorm(size = 5, alpha=0.0001, beta=0.75)
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # self.fc1 = nn.Linear(256*6*6, 4096)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 1000)

        # self.relu = nn.ReLU()

        #feature맵 생성
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernerl_size =11, stride = 4, padding=0),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=63, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.LocalResponseNorm(size = 5, alpha=0.0001, beta=0.75),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        
        # 분류기 
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
            
        self.init_bias()

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)


    def forward(self, x):
        x = self.net(x)
        x = x.view(-1,256*6*6)
        return self.classifier(x)        




