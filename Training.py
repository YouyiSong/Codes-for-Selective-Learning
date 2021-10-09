import torch
import numpy as np
import time
from Tools import DeviceInitialization
from Tools import DataSplit
from Tools import ExtraCT
from Tools import ExtraDataSingle
from Tools import ExtraDataMultiple
from torch.utils.data import DataLoader
from Dataset import DataSet
from Dataset import DataSetMerge
from Model import UNet
from Loss import DiceLoss
from Tools import weightOptim
from Metric import DSC
from Metric import AverageDSC
from Tools import csvWrite


############# Learning Setting ####################
set = 'BTCV'
path = 'D:\\DataSets\\UntrustedLearning\\Data\\'
modelPath = 'D:\\DataSets\\UntrustedLearning\\Model\\Ablation\\'
device = DeviceInitialization('cuda:0')
batch_size = 16
class_num = 9
learning_rate = 3e-4
fracTrain = 50
fracTest = 50
epoch_num = 40
criterion = DiceLoss()
xi = 0.05
xiRate = 1e-4
fracExtra = 50
para = 5

############# Reading Data ####################
modelName = 'Parameter_' + str(fracExtra) + '_' + str(para) + '_1'
TrainIdxTarget, TestIdx = DataSplit(path=path, set=set, fracTrain=fracTrain, fracTest=fracTest)
trainSetTarget = DataSet(dataPath=path, dataName=TrainIdxTarget, height=256, width=256)
testSet = DataSet(dataPath=path, dataName=TestIdx, height=256, width=256)
TrainSetTarget = torch.utils.data.DataLoader(dataset=trainSetTarget, batch_size=batch_size, shuffle=False, num_workers=0)
TestSet = torch.utils.data.DataLoader(dataset=testSet, batch_size=batch_size, shuffle=False, num_workers=0)
ExtraCTIdx = ExtraCT(path=path, set=set, frac=fracExtra)
cost = np.empty(len(ExtraCTIdx))

NetCon = UNet(num=class_num)
NetCon.load_state_dict(torch.load('D:\\DataSets\\UntrustedLearning\\Model\\TargetOnly\\Performance_BTCV_50_50_1.pkl'))
NetCon.to(device)
NetCon.eval()
Net = UNet(num=class_num)
Net.to(device)
optim = torch.optim.Adam(Net.parameters(), lr=learning_rate)

############# Model Training ####################
start_time = time.time()
IterNum = 0
for epoch in range(epoch_num):
    if epoch > 10:
        Net.eval()
        costMean = 0
        for idx, (images, targets) in enumerate(TrainSetTarget, 0):
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                outputs = Net(images)
            lossDice = criterion(outputs, targets)
            costMean += lossDice.mean()
        costMean /= idx

        for ii in range(len(ExtraCTIdx)):
            ExtraIdx = ExtraDataSingle(path=path, data=ExtraCTIdx[ii])
            extraSet = DataSet(dataPath=path, dataName=ExtraIdx, height=256, width=256)
            ExtraSet = torch.utils.data.DataLoader(dataset=extraSet, batch_size=batch_size, shuffle=False,
                                                   num_workers=0)
            tempCost = 0
            for idx, (images, targets) in enumerate(ExtraSet, 0):
                images = images.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    outputs = Net(images)
                lossDice = criterion(outputs, targets)
                tempCost += lossDice.mean()
            cost[ii] = abs(tempCost / idx - costMean)
        weight = weightOptim(cost, balance=para)
    else:
        weight = list(np.ones(len(ExtraCTIdx)))

    ExtraIdx, weight = ExtraDataMultiple(path, ExtraCTIdx, weight)
    weighTarget = list(np.zeros(len(TrainIdxTarget)))
    trainSet = DataSetMerge(dataPath=path, dataName=TrainIdxTarget + ExtraIdx, weight=weighTarget + weight, height=256,
                            width=256)
    TrainSet = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=0)

    Net.train()
    tempLoss = 0
    if epoch > 20:
        for idx, (images, targets, weight) in enumerate(TrainSet, 0):
            images = images.to(device)
            targets = targets.to(device)
            weight = weight.to(device)
            weightTarget = torch.where(weight == 0, torch.ones_like(weight), torch.zeros_like(weight))
            weight[weight == 0] = 1

            outputs = Net(images)
            lossDice = criterion(outputs, targets)

            with torch.no_grad():
                outputsTemp = NetCon(images)
            lossCon = criterion(outputsTemp, targets)

            lossCon = (lossDice - lossCon) * weightTarget
            lossCon = lossCon[lossCon > 0]
            lossCon = lossCon.sum()

            loss = (weight * lossDice).mean() + xi * lossCon
            optim.zero_grad()
            loss.backward()
            optim.step()
            xi += xiRate * lossCon.detach()
            tempLoss += loss
        IterNum += (idx + 1)
        print("Epoch:%02d  ||  Iteration:%04d  ||  Loss:%.4f  ||  Time elapsed:%.2f(min)"
              % (epoch + 1, IterNum, tempLoss / (idx + 1), (time.time() - start_time) / 60))
    else:
        for idx, (images, targets, weight) in enumerate(TrainSet, 0):
            images = images.to(device)
            targets = targets.to(device)
            weight = weight.to(device)
            weight[weight == 0] = 1

            outputs = Net(images)
            lossDice = criterion(outputs, targets)
            loss = (weight * lossDice).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            tempLoss += loss
        IterNum += (idx + 1)
        print("Epoch:%02d  ||  Iteration:%04d  ||  Loss:%.4f  ||  Time elapsed:%.2f(min)"
              % (epoch + 1, IterNum, tempLoss / (idx + 1), (time.time() - start_time) / 60))

    if (epoch + 1) > 20:
        Net.eval()
        Res = []
        torch.set_printoptions(precision=2, sci_mode=False)
        for idx, (images, targets) in enumerate(TestSet, 0):
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                outputs = Net(images)
            _, segIdx = torch.max(outputs, dim=1)
            for ii in range(class_num):
                outputs[:, ii] = torch.where(segIdx == ii, torch.ones_like(segIdx), torch.zeros_like(segIdx))
            tempDSC = DSC(outputs, targets)
            Res.append(tempDSC)
        Res = torch.cat(Res, dim=0).cpu()
        Aver = 100 * AverageDSC(Res)
        csvWrite(path=modelPath, name=modelName, data=Aver, epoch=epoch + 1)
torch.save(Net.state_dict(), modelPath + modelName + '.pkl')
