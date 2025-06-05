"""데이터 준비 스크립트."""

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

# 저장 경로 생성 함수
def ensure_directories(paths):
    """주어진 경로 목록을 모두 생성한다."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


ensure_directories([
    DeepLearningPath,
    Path360To720,
    Path180To720,
    Path90To720,
])

for base in (Path360To720, Path180To720, Path90To720):
    ensure_directories([
        base + AllFolderName,
        base + TestFolderName,
        base + ValidationFolderName,
        base + TrainFolderName,
    ])

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

def split_dataset():
    """학습, 검증, 테스트 데이터셋을 분리한다."""
    lst_label = [f for f in os.listdir(Path360To720 + AllFolderName) if f.endswith("_Label.raw")]
    np.random.shuffle(lst_label)
    n_total = len(lst_label)
    n_test = int(n_total * 0.1)
    n_validation = int(n_total * 0.2)
    lst_test = lst_label[:n_test]
    lst_validation = lst_label[n_test:n_test + n_validation]
    lst_train = lst_label[n_test + n_validation:]

    def _copy_files(file_list, dst_folder):
        for name in file_list:
            shutil.copy(Path360To720 + AllFolderName + name.replace("_Label.raw", "_Input.raw"), dst_folder + name.replace("_Label.raw", "_Input.raw"))
            shutil.copy(Path360To720 + AllFolderName + name, dst_folder + name)
            shutil.copy(Path180To720 + AllFolderName + name.replace("_Label.raw", "_Input.raw"), dst_folder.replace(Path360To720, Path180To720) + name.replace("_Label.raw", "_Input.raw"))
            shutil.copy(Path90To720 + AllFolderName + name.replace("_Label.raw", "_Input.raw"), dst_folder.replace(Path360To720, Path90To720) + name.replace("_Label.raw", "_Input.raw"))
            shutil.copy(Path180To720 + AllFolderName + name, dst_folder.replace(Path360To720, Path180To720) + name)
            shutil.copy(Path90To720 + AllFolderName + name, dst_folder.replace(Path360To720, Path90To720) + name)

    _copy_files(lst_test, Path360To720 + TestFolderName)
    _copy_files(lst_validation, Path360To720 + ValidationFolderName)
    _copy_files(lst_train, Path360To720 + TrainFolderName)


def main():
    split_dataset()


if __name__ == "__main__":
    main()
