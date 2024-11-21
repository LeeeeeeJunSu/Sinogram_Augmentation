import os
import time
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from os import listdir
from os.path import join
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

from Engine import CustomEngineUNet

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

engine = CustomEngineUNet(Path90To720)
engine.Predict()
