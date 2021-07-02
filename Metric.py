import torch


def DSC(outputs, targets):
    batchSize = outputs.size(0)
    classNum = outputs.size(1)
    Res = torch.empty(batchSize, classNum)
    for ii in range(batchSize):
        for jj in range(classNum):
            tempTarget = targets[ii, jj]
            if tempTarget.sum() > 0:
                tempOutput = outputs[ii, jj]
                inter = tempTarget * tempOutput
                Res[ii, jj] = 2. * inter.sum() / (tempOutput.sum() + tempTarget.sum())
            else:
                Res[ii, jj] = -1
    return Res


def AverageDSC(data):
    ObjNum = data.size(1)
    Res = torch.empty(ObjNum+1)
    for ii in range(ObjNum):
        temp = data[:, ii]
        temp = temp[temp > -1]
        if temp.size(0) > 0:
            Res[ii] = temp.mean()
        else:
            Res[ii] = 0
    Res[ObjNum] = Res[:ObjNum].mean()
    return Res
