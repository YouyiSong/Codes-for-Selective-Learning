import numpy as np
import os
import nibabel as nib
from PIL import Image
import SimpleITK as sitk
import time


############# function space ####################
def NiiReading(path):
   niiData = nib.load(path+'.nii')
   return niiData.get_fdata()

def DcmReading(path):
   dcmReader = sitk.ImageSeriesReader()
   dcms = dcmReader.GetGDCMSeriesIDs(path)
   dcms = dcmReader.GetGDCMSeriesFileNames(path, dcms[0])
   dcmReader.SetFileNames(dcms)
   Data = dcmReader.Execute()
   return sitk.GetArrayFromImage(Data).astype(np.float)

def IntensityNormalization(img, min, max):
   img = np.where(img < min, min, img)
   img = np.where(img > max, max, img)
   min = np.min(img)
   max = np.max(img)
   return 255 * (img - min) / (max - min)

def AngleTransfrom(img, mask, set):
   mask = np.flip(mask, 0)
   if set == 'BTCV':
      img = np.flip(img, 0)
      img = Image.fromarray(img)
      img = img.rotate(90)
   elif set == 'TCIA':
      img = Image.fromarray(img)
      img = img.rotate(180)
   mask = Image.fromarray(mask)
   mask = mask.rotate(90)
   mask = np.asarray(mask)
   img = np.asarray(img)
   return img, mask

def ColorMapping(mask):
    ####index#########
    #   object    mask      color
    # Duodenum     14   [255, 0, 0]
    # Esophagus    5    [0, 255, 0]
    # Gallbladder  4    [0, 0, 255]
    # Liver        6    [255, 255, 0]
    # L-Kidney     3    [255, 0, 255]
    # Pancreas     11   [0, 255, 255]
    # Spleen       1    [255, 255, 255]
    # Stomach      7    [128, 128, 128]
    out = np.zeros((mask.shape[0], mask.shape[1], 3))
    out[:, :, 0] += np.where(mask == 14, 255, 0) # Duodenum
    out[:, :, 1] += np.where(mask == 5, 255, 0)  # Esophagus
    out[:, :, 2] += np.where(mask == 4, 255, 0)  # Gallbladder
    out[:, :, 0] += np.where(mask == 6, 255, 0)  # Liver
    out[:, :, 1] += np.where(mask == 6, 255, 0)
    out[:, :, 0] += np.where(mask == 3, 255, 0)  # L-Kidney
    out[:, :, 2] += np.where(mask == 3, 255, 0)
    out[:, :, 1] += np.where(mask == 11, 255, 0) # Pancreas
    out[:, :, 2] += np.where(mask == 11, 255, 0)
    out[:, :, 0] += np.where(mask == 1, 255, 0)  # Spleen
    out[:, :, 1] += np.where(mask == 1, 255, 0)
    out[:, :, 2] += np.where(mask == 1, 255, 0)
    out[:, :, 0] += np.where(mask == 7, 128, 0)  # Stomach
    out[:, :, 1] += np.where(mask == 7, 128, 0)
    out[:, :, 2] += np.where(mask == 7, 128, 0)
    return out


def CsvWrite(mask, set, path, niiIdx, imgIdx):
   objName = 'Spleen'
   objIdx = 1
   if np.where(mask == objIdx)[0].size > 0:
      write(set, path, objName, niiIdx, imgIdx)
   objName = 'L-Kidney'
   objIdx = 3
   if np.where(mask == objIdx)[0].size > 0:
      write(set, path, objName, niiIdx, imgIdx)
   objName = 'Gallbladder'
   objIdx = 4
   if np.where(mask == objIdx)[0].size > 0:
      write(set, path, objName, niiIdx, imgIdx)
   objName = 'Esophagus'
   objIdx = 5
   if np.where(mask == objIdx)[0].size > 0:
      write(set, path, objName, niiIdx, imgIdx)
   objName = 'Liver'
   objIdx = 6
   if np.where(mask == objIdx)[0].size > 0:
      write(set, path, objName, niiIdx, imgIdx)
   objName = 'Stomach'
   objIdx = 7
   if np.where(mask == objIdx)[0].size > 0:
      write(set, path, objName, niiIdx, imgIdx)
   objName = 'Pancreas'
   objIdx = 11
   if np.where(mask == objIdx)[0].size > 0:
      write(set, path, objName, niiIdx, imgIdx)
   objName = 'Duodenum'
   objIdx = 14
   if np.where(mask == objIdx)[0].size > 0:
      write(set, path, objName, niiIdx, imgIdx)


def write(set, path, objName, niiIdx, imgIdx):
   outname = '%04d'%imgIdx
   csvName = open('%s\\CSVs\\%s\\%s_%s%s' % (path, objName, set, niiIdx, '.txt'), 'a')
   csvName.write(outname + '\n')
   csvName.close()


dataPath = 'D:\\DataSets\\CTSegmentation\\Data'
outPath = 'D:\\DataSets\\SamplingLearning\\Data\\'

set = 'BTCV'
dataName = np.genfromtxt(os.path.join(dataPath, set, 'name.txt'), dtype=str)
start_time = time.time()
for ii in range(dataName.size):
   imgData = NiiReading(os.path.join(dataPath, set, 'Images', dataName[ii]))
   imgData = IntensityNormalization(imgData, -200, 250)
   maskData = NiiReading(os.path.join(dataPath, set, 'Labels', dataName[ii]))
   for jj in range(imgData.shape[2]):
       mask = maskData[:, :, jj]
       img = imgData[:, :, jj]
       img, mask = AngleTransfrom(img, mask, set=set)
       CsvWrite(mask, set='BTCV', path=outPath, niiIdx=dataName[ii], imgIdx=jj)
       mask = ColorMapping(mask)
       if np.sum(mask) > 0:
           img = Image.fromarray(img).convert('LA')
           mask = Image.fromarray(mask.astype(np.uint8)).convert('RGB')
           outName = '%04d' % jj
           csvName = open('%s\\CSVs\\All\\\%s_%s%s' % (outPath, set, dataName[ii], '.txt'), 'a')
           csvName.write(outName + '\n')
           csvName.close()
           img.save('%s\\Images\\%s_%s_%s.png' % (outPath, set, dataName[ii], outName))
           mask.save('%s\\Masks\\%s_%s_%s.png' % (outPath, set, dataName[ii], outName))
   print("BTCV %s has done with time elapsed %.2f (min)" % (dataName[ii], (time.time() - start_time) / 60))


set = 'TCIA'
dataName = np.genfromtxt(os.path.join(dataPath, set, 'name.txt'), dtype=str)
start_time = time.time()
for ii in range(dataName.size):
   imgData = DcmReading(os.path.join(dataPath, set, 'Images', dataName[ii]))
   imgData = IntensityNormalization(imgData, -200, 250)
   maskData = NiiReading(os.path.join(dataPath, set, 'Labels', 'label00' + dataName[ii]))
   for jj in range(imgData.shape[0]):
       mask = maskData[:, :, imgData.shape[0] - jj - 1]
       img = imgData[imgData.shape[0] - jj - 1, :, :]
       img, mask = AngleTransfrom(img, mask, set=set)
       CsvWrite(mask, set='TCIA', path=outPath, niiIdx=dataName[ii], imgIdx=jj)
       mask = ColorMapping(mask)
       if np.sum(mask) > 0:
           img = Image.fromarray(img).convert('LA')
           mask = Image.fromarray(mask.astype(np.uint8)).convert('RGB')
           outName = '%04d' % jj
           csvName = open('%s\\CSVs\\All\\\%s_%s%s' % (outPath, set, dataName[ii], '.txt'), 'a')
           csvName.write(outName + '\n')
           csvName.close()
           img.save('%s\\Images\\%s_%s_%s.png' % (outPath, set, dataName[ii], outName))
           mask.save('%s\\Masks\\%s_%s_%s.png' % (outPath, set, dataName[ii], outName))
   print("TCIA %s has done with time elapsed %.2f (min)" % (dataName[ii], (time.time() - start_time) / 60))