import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import OxfordIIITPet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ViT import VisionTransformer
from torchvision.transforms import Resize, ToTensor

to_tensor = [Resize((144,144)), ToTensor()]




EPOCH = 1000
device = "cuda"

model = VisionTransformer.to(device=device)
opt = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss(reduction="sum")

#각각이 무슨 의미이며 무슨 역할을 할까?
for epoch in range(EPOCH):
    epoch_loss = []
    model.train()
    # for step, (inputs, labels) in enumerate(train_data_loader):





 