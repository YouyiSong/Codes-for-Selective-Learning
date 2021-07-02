import numpy as np
import os
import torch
import random
import math
from scipy.optimize import minimize
from scipy.optimize import Bounds
import pulp as plp


def DeviceInitialization(GPUNum):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = torch.device(GPUNum)
    else:
        device = torch.device('cpu')

    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    return device


def DataSplit(path, set, fracTrain, fracTest):
    trainIdx = []
    testIdx = []

    data = np.genfromtxt(path + 'CSVs\\' + set + '.txt', dtype=str)
    shuffleIdx = np.arange(len(data))
    shuffleRng = np.random.RandomState(2021)
    shuffleRng.shuffle(shuffleIdx)
    data = data[shuffleIdx]
    TrainNum = math.ceil(fracTrain * len(data) / 100)
    TestNum = len(data) - math.ceil(fracTest * len(data) / 100)
    train = data[:TrainNum]
    test = data[TestNum:]

    for ii in range(len(train)):
        trainIdx.append(set + '_' + train[ii])
    for ii in range(len(test)):
        testIdx.append(set + '_' + test[ii])

    TrainIdx = []
    TestIdx = []

    for ii in range(len(trainIdx)):
        data = trainIdx[ii]
        file = path + 'CSVs\\All\\' + data + '.txt'
        if os.path.isfile(file):
            dataSlice = np.genfromtxt(file, dtype=str)
            for jj in range(len(dataSlice)):
                temp = data + '_' + dataSlice[jj]
                TrainIdx.append(temp)

    for ii in range(len(testIdx)):
        data = testIdx[ii]
        file = path + 'CSVs\\All\\' + data + '.txt'
        if os.path.isfile(file):
            dataSlice = np.genfromtxt(file, dtype=str)
            for jj in range(len(dataSlice)):
                temp = data + '_' + dataSlice[jj]
                TestIdx.append(temp)

    return TrainIdx, TestIdx


def ExtraData(path, set, frac):
    if set == 'BTCV':
        set = 'TCIA'
    else:
        set = 'BTCV'
    dataIdx = []
    data = np.genfromtxt(path + 'CSVs\\' + set + '.txt', dtype=str)
    shuffleIdx = np.arange(len(data))
    shuffleRng = np.random.RandomState(2021)
    shuffleRng.shuffle(shuffleIdx)
    data = data[shuffleIdx]
    dataNum = math.ceil(frac * len(data) / 100)
    extra = data[:dataNum]

    for ii in range(len(extra)):
        dataIdx.append(set + '_' + extra[ii])

    DataIdx = []
    for ii in range(len(dataIdx)):
        data = dataIdx[ii]
        file = path + 'CSVs\\All\\' + data + '.txt'
        if os.path.isfile(file):
            dataSlice = np.genfromtxt(file, dtype=str)
            for jj in range(len(dataSlice)):
                temp = data + '_' + dataSlice[jj]
                DataIdx.append(temp)

    return DataIdx


def ExtraCT(path, set, frac):
    if set == 'BTCV':
        set = 'TCIA'
    else:
        set = 'BTCV'
    dataIdx = []
    data = np.genfromtxt(path + 'CSVs\\' + set + '.txt', dtype=str)
    shuffleIdx = np.arange(len(data))
    shuffleRng = np.random.RandomState(2021)
    shuffleRng.shuffle(shuffleIdx)
    data = data[shuffleIdx]
    dataNum = math.ceil(frac * len(data) / 100)
    extra = data[:dataNum]

    for ii in range(len(extra)):
        dataIdx.append(set + '_' + extra[ii])

    return dataIdx


def ExtraDataSingle(path, data):
    DataIdx = []
    file = path + 'CSVs\\All\\' + data + '.txt'
    if os.path.isfile(file):
        dataSlice = np.genfromtxt(file, dtype=str)
        for jj in range(len(dataSlice)):
            temp = data + '_' + dataSlice[jj]
            DataIdx.append(temp)

    return DataIdx


def ExtraDataMultiple(path, dataIdx, weight):
    DataIdx = []
    imageWeight = []
    for ii in range(len(dataIdx)):
        if weight[ii] > 0:
            data = dataIdx[ii]
            file = path + 'CSVs\\All\\' + data + '.txt'
            if os.path.isfile(file):
                dataSlice = np.genfromtxt(file, dtype=str)
                for jj in range(len(dataSlice)):
                    temp = data + '_' + dataSlice[jj]
                    DataIdx.append(temp)
                    imageWeight.append(weight[ii])

    return DataIdx, imageWeight


def csvWrite(path, name, data, epoch):
    csvName = open('%s\\%s.txt' % (path, name), 'a')
    csvName.write(str(epoch) + ':\t')
    for ii in range(len(data)):
        outData = '%.2f' % (data[ii])
        csvName.write(str(outData) + '\t')
    csvName.write('\n')
    csvName.close()


def costSetting(x):
    global cost
    cost = x


def balanceSetting(x):
    global balance
    balance = x


def obj(w):
    return (w*cost).sum() + balance*(w*w).sum()


def constraint(w):
    return w.sum() - 1


def weightOptim(cost, balance):
    costSetting(cost)
    balanceSetting(balance)
    w0 = np.ones_like(cost) / cost.shape[0]
    bound = Bounds(np.zeros_like(cost), np.ones_like(cost))
    weight = minimize(obj, w0, method='SLSQP', tol=1e-6, bounds=bound,
                      constraints={'fun': constraint, 'type': 'eq'})
    weight = weight.x

    return weight*len(weight)


def dataSelect(data, fraction):
    lpModel = plp.LpProblem("Weight Minimum", sense=plp.LpMinimize)
    weightItem = np.arange(data.shape[0])
    vars = plp.LpVariable.dicts("weight", weightItem, lowBound=0, upBound=1, cat=plp.LpContinuous)
    lpModel += plp.lpSum([data[ii] * vars[ii] for ii in weightItem])
    lpModel += (plp.lpSum([vars[ii] for ii in weightItem]) == fraction * len(data))
    lpModel.solve(plp.PULP_CBC_CMD(msg=0))
    weight = []
    for var in lpModel.variables():
        weight.append(var.value())
    weight = np.array(weight)

    return weight

