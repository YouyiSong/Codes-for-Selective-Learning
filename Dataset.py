import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class DataSet(torch.utils.data.Dataset):
    def __init__(self, dataPath, dataName, width, height):
        super(DataSet, self).__init__()
        self.path = dataPath
        self.name = dataName
        self.width = width
        self.height = height
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        img = Image.open(self.path + 'Images\\' + self.name[idx] + '.png')
        img = np.asarray(img.resize((self.height, self.width), Image.NEAREST))
        img = np.array(img[:, :, 0])

        mask = Image.open(self.path + 'Masks\\' + self.name[idx] + '.png')
        mask = np.asarray(mask.resize((self.height, self.width), Image.NEAREST))

        ###Image to Segmentation Map##############
        maskDuo = np.where((mask[:, :, 0] == 255) &
                           (mask[:, :, 1] == 0) &
                           (mask[:, :, 2] == 0), 1, 0)

        maskEso = np.where((mask[:, :, 0] == 0) &
                           (mask[:, :, 1] == 255) &
                           (mask[:, :, 2] == 0), 1, 0)

        maskGal = np.where((mask[:, :, 0] == 0) &
                           (mask[:, :, 1] == 0) &
                           (mask[:, :, 2] == 255), 1, 0)

        maskLiv = np.where((mask[:, :, 0] == 255) &
                           (mask[:, :, 1] == 255) &
                           (mask[:, :, 2] == 0), 1, 0)

        maskLKi = np.where((mask[:, :, 0] == 255) &
                           (mask[:, :, 1] == 0) &
                           (mask[:, :, 2] == 255), 1, 0)

        maskPan = np.where((mask[:, :, 0] == 0) &
                           (mask[:, :, 1] == 255) &
                           (mask[:, :, 2] == 255), 1, 0)

        maskSpl = np.where((mask[:, :, 0] == 255) &
                           (mask[:, :, 1] == 255) &
                           (mask[:, :, 2] == 255), 1, 0)

        maskSto = np.where((mask[:, :, 0] == 128) &
                           (mask[:, :, 1] == 128) &
                           (mask[:, :, 2] == 128), 1, 0)

        maskBac = np.where((mask[:, :, 0] == 0) &
                           (mask[:, :, 1] == 0) &
                           (mask[:, :, 2] == 0), 1, 0)

        mask = np.dstack((maskDuo, maskEso, maskGal, maskLiv, maskLKi, maskPan, maskSpl, maskSto,  maskBac))
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask


class DataSetMerge(torch.utils.data.Dataset):
    def __init__(self, dataPath, dataName, weight, width, height):
        super(DataSetMerge, self).__init__()
        self.path = dataPath
        self.name = dataName
        self.weight = weight
        self.width = width
        self.height = height
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        img = Image.open(self.path + 'Images\\' + self.name[idx] + '.png')
        img = np.asarray(img.resize((self.height, self.width), Image.NEAREST))
        img = np.array(img[:, :, 0])

        mask = Image.open(self.path + 'Masks\\' + self.name[idx] + '.png')
        mask = np.asarray(mask.resize((self.height, self.width), Image.NEAREST))

        ###Image to Segmentation Map##############
        maskDuo = np.where((mask[:, :, 0] == 255) &
                           (mask[:, :, 1] == 0) &
                           (mask[:, :, 2] == 0), 1, 0)

        maskEso = np.where((mask[:, :, 0] == 0) &
                           (mask[:, :, 1] == 255) &
                           (mask[:, :, 2] == 0), 1, 0)

        maskGal = np.where((mask[:, :, 0] == 0) &
                           (mask[:, :, 1] == 0) &
                           (mask[:, :, 2] == 255), 1, 0)

        maskLiv = np.where((mask[:, :, 0] == 255) &
                           (mask[:, :, 1] == 255) &
                           (mask[:, :, 2] == 0), 1, 0)

        maskLKi = np.where((mask[:, :, 0] == 255) &
                           (mask[:, :, 1] == 0) &
                           (mask[:, :, 2] == 255), 1, 0)

        maskPan = np.where((mask[:, :, 0] == 0) &
                           (mask[:, :, 1] == 255) &
                           (mask[:, :, 2] == 255), 1, 0)

        maskSpl = np.where((mask[:, :, 0] == 255) &
                           (mask[:, :, 1] == 255) &
                           (mask[:, :, 2] == 255), 1, 0)

        maskSto = np.where((mask[:, :, 0] == 128) &
                           (mask[:, :, 1] == 128) &
                           (mask[:, :, 2] == 128), 1, 0)

        maskBac = np.where((mask[:, :, 0] == 0) &
                           (mask[:, :, 1] == 0) &
                           (mask[:, :, 2] == 0), 1, 0)

        mask = np.dstack((maskDuo, maskEso, maskGal, maskLiv, maskLKi, maskPan, maskSpl, maskSto,  maskBac))
        img = self.transform(img)
        mask = self.transform(mask)
        weight = self.weight[idx]
        return img, mask, weight