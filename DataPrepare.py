import os
import numpy as np
import shutil

RootPath = "E:\\Data\\SinogramAugmentation\\Data\\"
lstVolume = ["Shepplogan", "CylinderInCylinder", "PillarInPillar", "PillarInCylinder", "CylinderInPillar", "FourCylinder", "FourPillar", "TwoCylinderTwoPillar", "FourCylinderInPillar", "FourPillarInCylinder"]
LabelName = "Sinogram_Original720_512x720x512.raw"
Input360Name = "Sinogram_Linear360_512x720x512.raw"
Input180Name = "Sinogram_Linear180_512x720x512.raw"
Input90Name = "Sinogram_Linear90_512x720x512.raw"
DeepLearningPath = RootPath + "DeepLearning\\"
Path360To720 = DeepLearningPath + "360To720\\"
Path180To720 = DeepLearningPath + "180To720\\"
Path90To720 = DeepLearningPath + "90To720\\"
AllFolderName = "All\\"
TestFolderName = "Test\\"
ValidationFolderName = "Validation\\"
TrainFolderName = "Train\\"

# 저장 경로 생성
if not os.path.exists(DeepLearningPath):
    os.makedirs(DeepLearningPath)
if not os.path.exists(Path360To720):
    os.makedirs(Path360To720)
if not os.path.exists(Path180To720):
    os.makedirs(Path180To720)
if not os.path.exists(Path90To720):
    os.makedirs(Path90To720)
if not os.path.exists(Path360To720 + AllFolderName):
    os.makedirs(Path360To720 + AllFolderName)
if not os.path.exists(Path360To720 + TestFolderName):
    os.makedirs(Path360To720 + TestFolderName)
if not os.path.exists(Path360To720 + ValidationFolderName):
    os.makedirs(Path360To720 + ValidationFolderName)
if not os.path.exists(Path360To720 + TrainFolderName):
    os.makedirs(Path360To720 + TrainFolderName)
if not os.path.exists(Path180To720 + AllFolderName):
    os.makedirs(Path180To720 + AllFolderName)
if not os.path.exists(Path180To720 + TestFolderName):
    os.makedirs(Path180To720 + TestFolderName)
if not os.path.exists(Path180To720 + ValidationFolderName):
    os.makedirs(Path180To720 + ValidationFolderName)
if not os.path.exists(Path180To720 + TrainFolderName):
    os.makedirs(Path180To720 + TrainFolderName)
if not os.path.exists(Path90To720 + AllFolderName):
    os.makedirs(Path90To720 + AllFolderName)
if not os.path.exists(Path90To720 + TestFolderName):
    os.makedirs(Path90To720 + TestFolderName)
if not os.path.exists(Path90To720 + ValidationFolderName):
    os.makedirs(Path90To720 + ValidationFolderName)
if not os.path.exists(Path90To720 + TrainFolderName):
    os.makedirs(Path90To720 + TrainFolderName)

'''
for volName in lstVolume:
    LabelPath = RootPath + volName + "\\" + LabelName
    Input360Path = RootPath + volName + "\\" + Input360Name
    Input180Path = RootPath + volName + "\\" + Input180Name
    Input90Path = RootPath + volName + "\\" + Input90Name
    LabelData = np.fromfile(LabelPath, dtype=np.float32)
    Input360Data = np.fromfile(Input360Path, dtype=np.float32)
    Input180Data = np.fromfile(Input180Path, dtype=np.float32)
    Input90Data = np.fromfile(Input90Path, dtype=np.float32)
    LabelData = LabelData.reshape([512, 720, 512])
    Input360Data = Input360Data.reshape([512, 720, 512])
    Input180Data = Input180Data.reshape([512, 720, 512])
    Input90Data = Input90Data.reshape([512, 720, 512])
    for i in range(512):
        LabelSlice = LabelData[i, :, :]
        Input360Slice = Input360Data[i, :, :]
        Input180Slice = Input180Data[i, :, :]
        Input90Slice = Input90Data[i, :, :]
        if np.mean(LabelSlice) >= 1:
            Input360Slice.tofile(Path360To720 + AllFolderName + volName + "_" + str(i) + "_Input.raw")
            LabelSlice.tofile(Path360To720 + AllFolderName + volName + "_" + str(i) + "_Label.raw")
            Input180Slice.tofile(Path180To720 + AllFolderName + volName + "_" + str(i) + "_Input.raw")
            LabelSlice.tofile(Path180To720 + AllFolderName + volName + "_" + str(i) + "_Label.raw")
            Input90Slice.tofile(Path90To720 + AllFolderName + volName + "_" + str(i) + "_Input.raw")
            LabelSlice.tofile(Path90To720 + AllFolderName + volName + "_" + str(i) + "_Label.raw")
        else:
            print(volName + " " + str(i) + " is skipped - Label Mean Intensity is " + str(np.mean(LabelSlice)))
'''

lstLabel = os.listdir(Path360To720 + AllFolderName)
lstLabel = [f for f in lstLabel if f.endswith("_Label.raw")]
np.random.shuffle(lstLabel)
nTotal = len(lstLabel)
nTest = int(nTotal * 0.1)
nValidation = int(nTotal * 0.2)
nTrain = nTotal - nTest - nValidation
lstTest = lstLabel[:nTest]
lstValidation = lstLabel[nTest:nTest+nValidation]
lstTrain = lstLabel[nTest+nValidation:]
for Test in lstTest:
    shutil.copy(Path360To720 + AllFolderName + Test.replace("_Label.raw", "_Input.raw"), Path360To720 + TestFolderName + Test.replace("_Label.raw", "_Input.raw"))
    shutil.copy(Path180To720 + AllFolderName + Test.replace("_Label.raw", "_Input.raw"), Path180To720 + TestFolderName + Test.replace("_Label.raw", "_Input.raw"))
    shutil.copy(Path90To720 + AllFolderName + Test.replace("_Label.raw", "_Input.raw"), Path90To720 + TestFolderName + Test.replace("_Label.raw", "_Input.raw"))
    shutil.copy(Path360To720 + AllFolderName + Test, Path360To720 + TestFolderName + Test)
    shutil.copy(Path360To720 + AllFolderName + Test, Path180To720 + TestFolderName + Test)
    shutil.copy(Path360To720 + AllFolderName + Test, Path90To720 + TestFolderName + Test)
for Validation in lstValidation:
    shutil.copy(Path360To720 + AllFolderName + Validation.replace("_Label.raw", "_Input.raw"), Path360To720 + ValidationFolderName + Validation.replace("_Label.raw", "_Input.raw"))
    shutil.copy(Path180To720 + AllFolderName + Validation.replace("_Label.raw", "_Input.raw"), Path180To720 + ValidationFolderName + Validation.replace("_Label.raw", "_Input.raw"))
    shutil.copy(Path90To720 + AllFolderName + Validation.replace("_Label.raw", "_Input.raw"), Path90To720 + ValidationFolderName + Validation.replace("_Label.raw", "_Input.raw"))
    shutil.copy(Path360To720 + AllFolderName + Validation, Path360To720 + ValidationFolderName + Validation)
    shutil.copy(Path360To720 + AllFolderName + Validation, Path180To720 + ValidationFolderName + Validation)
    shutil.copy(Path360To720 + AllFolderName + Validation, Path90To720 + ValidationFolderName + Validation)
for Train in lstTrain:
    shutil.copy(Path360To720 + AllFolderName + Train.replace("_Label.raw", "_Input.raw"), Path360To720 + TrainFolderName + Train.replace("_Label.raw", "_Input.raw"))
    shutil.copy(Path180To720 + AllFolderName + Train.replace("_Label.raw", "_Input.raw"), Path180To720 + TrainFolderName + Train.replace("_Label.raw", "_Input.raw"))
    shutil.copy(Path90To720 + AllFolderName + Train.replace("_Label.raw", "_Input.raw"), Path90To720 + TrainFolderName + Train.replace("_Label.raw", "_Input.raw"))
    shutil.copy(Path360To720 + AllFolderName + Train, Path360To720 + TrainFolderName + Train)
    shutil.copy(Path360To720 + AllFolderName + Train, Path180To720 + TrainFolderName + Train)
    shutil.copy(Path360To720 + AllFolderName + Train, Path90To720 + TrainFolderName + Train)