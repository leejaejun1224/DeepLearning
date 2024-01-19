import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
from model import Alexnet

"""
dataset 다운로드 필요함.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227
NUM_CLASS = 1000
DEVICE_IDS = [0,1,2,3]

INPUT_ROOT_DIR = ''
TRAIN_IMG_DIR = ''
OUTPUT_DIR = ''
LOG_DIR = OUTPUT_DIR + ''
CHECK_POINT_DIR = ''

def main():
    seed = torch.initial_seed()
    alexnet = Alexnet(num_classes=NUM_CLASS)

    dataset = datasets.ImageFolder(TRAIN_IMG_DIR,
                                   transforms.Compose([transforms.CenterCrop(IMAGE_DIM), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])]))

    dataloader = data.DataLoader(dataset=dataset,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=8,
                                 drop_last=True,
                                 batch_size=BATCH_SIZE)

    optimizer = optim.SGD(params=alexnet.parameters(), 
                          lr = LR_INIT,
                          momentum=MOMENTUM,
                          weight_decay=LR_DECAY)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

    total_steps = 1

    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()
        for imgs, classes in dataloader:
            imgs, classes  = imgs.to(device), classes.to(device)
            output = alexnet(imgs)

            loss = F.cross_entropy(output, classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    print("done")


if __name__=="__main__":
    main()