import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        Dice = DSC()
        classNum = target.size(1)
        loss = Dice(output[:, 1], target[:, 1])
        for ii in range(classNum):
            loss += Dice(output[:, ii], target[:, ii])
        return loss


class DSC(nn.Module):
    def __init__(self):
        super(DSC, self).__init__()

    def forward(self, output, target):
        batchSize = target.size(0)
        target = target.view(batchSize, -1)
        output = output.view(batchSize, -1)
        inter = (output * target).sum(1)
        union = output.sum(1) + target.sum(1) + 1e-10
        loss = 1 - (2. * inter + 1e-10) / union
        return loss