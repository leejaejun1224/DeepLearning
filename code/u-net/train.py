import PIL
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

import os
import torchvision.transforms as transforms

label_class = {(0,0,0):255, (111,74,0):255, (81,0,81):255, (128,64,128):0, (224,35,232):1, (250,170,160):255, (230,150,140):255, 
               (70,70,70):2, (102,102,156):3, (190,153,153):4, (180,165,180):255, (150,100,100):255, (150,120,90):255,
               (153,153,153):5, (250,170,30):6, (220,220,0):7, (107,142,35):8, (152,251,152):9, (70,130,180):10,
               (220,20,60):11, (255,0,0):12, (0,0,142):13, (0,0,70):14, (0,60,100):13, (0,0,90):255,
               (0,0,110):255, (0,80,100):16, (0,0,230):17, (119,11,32):18}

baseline = 0.54

def get_transform(method=Image.BICUBIC, normalize=True):      
    transform_list = []
    osize = [1242,375]
    transform_list.append(transforms.Resize(osize, method))
    transform_list += [transforms.ToTensor()]
    if normalize:
        # 여기서 normalize하는 것에 대해 더 살펴보면
        # [0.5,0.5,0.5]는 각각 RGB에 대한 것이고 앞의 [0.5,0.5,0.5]는 RGB의 값의 평균을 모두 0.5로, 뒤의 것은 표준편차가 RGB 모두 0.5로
        # 맞춘 것이다. 긍까 0부터1로 되어있는 픽셀의 값에 0.5를 빼고 이를 0.5로 나누어서 -1~1의 사이의 값으로 만드는 것이다.
        transform_list += [transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
    return transforms.Compose(transform_list)

class KITTI_Dataloader(Dataset):
    __left = []
    __disp = []
    __seg = []
    
    def __init__(self):
        self.img_root = "/container/home/datasets/kitti_segmentation/training/image_2"
        self.mask_root = "/container/home/datasets/kitti_segmentation/training/semantic"
        self.disp_root = "/container/home/datasets/kitti_segmentation/training/image_2"

        for line in os.listdir(self.mask_root):
            self.__left.append(self.img_root + line)
            self.__seg.append(self.mask_root + line)

    def __getitem__(self, index):
        img = Image.open(self.__left[index]).convert("RGB")
        mask = Image.open(self.__seg[index])

        transform = get_transform()
        img = transform(img)

        mask = mask.resize((1242,375), PIL.Image.NEAREST)
        mask = np.array(mask)

        one_hot_pred = np.zeros((375,1242))

        for j in range(375):
            for k in range(1242):
                one_hot_pred[j][k] = label_class[tuple(mask[j][k])]

        one_hot_pred = torch.from_numpy(one_hot_pred).long()

        input_dict = {"img":img, "mask":one_hot_pred}

        return input_dict
    
    def __len__(self):
        return len(self.__left)



if __name__=="__main__":
    train_dataset = KITTI_Dataloader()

    train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=8)



