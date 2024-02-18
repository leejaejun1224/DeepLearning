import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop



class Unet(nn.Module):
    def __init__(self, n_class):
        super(Unet, self).__init__()
        self.n_class = n_class

        def contract(in_channel, out_channel, kernel_size, stride, padding):
            layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channel), 
                nn.ReLU()
            )
            
            return layer

        self.conv1 = nn.Sequential(
            contract(3,64,3,1,0),
            contract(64,64,3,1,0)
            )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            contract(64,128,3,1,0),
            contract(128,128,3,1,0)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            contract(128,256,3,1,0),
            contract(256,256,3,1,0)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            contract(256,512,3,1,0),
            contract(512,512,3,1,0)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
            contract(512,1024,3,1,0),
            contract(1024,1024,3,1,0)
        )

        self.unpool4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)

        self.upconv4 = nn.Sequential(
            contract(1024, 512,3,1,0),
            contract(512,512,3,1,0)
        )

        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)

        self.upconv3 = nn.Sequential(
            contract(512, 256, 3,1,0),
            contract(256,256, 3,1,0)
        )

        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)

        self.upconv2 = nn.Sequential(
            contract(256,128,3,1,0),
            contract(128,128,3,1,0)
        )

        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)

        self.upconv1 = nn.Sequential(
            contract(128,64,3,1,0),
            contract(64,64,3,1,0),
            contract(64,self.n_class, 3,1,0)
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)

        upconv4 = self.unpool4(conv5)
        unpool4 = self.upconv4(torch.cat([CenterCrop(upconv4.size(-2), upconv4.size(-1))(conv4), upconv4], dim=1))

        upconv3 = self.unpool3(unpool4)
        unpool3 = self.upconv3(torch.cat([CenterCrop(upconv3.size(-2), upconv3.size(-1))(conv3), upconv3], dim=1))

        upconv2 = self.unpool2(unpool3)
        unpool2 = self.upconv2(torch.cat([CenterCrop(upconv2.size(-1), upconv2.size(-1))(conv2), upconv2], dim=1))

        upconv1 = self.unpool1(unpool2)
        output = self.upconv1(torch.cat([CenterCrop(upconv1.size(-2), upconv2.size(-1))(conv1),upconv1], dim=1))

        return output

