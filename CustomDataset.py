import torch
from tifffile import imread
import numpy as np
import os
import torch.utils.data as data
    
class Dataset_Train(data.Dataset):
    def __init__(self, image_dir, delete_zero = True):
        super(Dataset_Train, self).__init__()
        self.path = os.path.join(image_dir) 
        self.imageNames = sorted([f for f in os.listdir(self.path) if f.endswith('.tif')])
        self.inputs = []
        self.labels = []
        for i in self.imageNames:
            filename = os.path.join(self.path, i)
            oneImage = imread(filename)
            for j in range(oneImage.shape[0] - 2):
                inputImage = oneImage[j:j+3,  :].copy()
                labelImage = inputImage.copy()
                inputImage[1, :] = 0
                if delete_zero and np.mean(inputImage) < 1.0:
                    continue
                self.inputs.append(torch.tensor(inputImage, dtype=torch.float32).unsqueeze(0))
                self.labels.append(torch.tensor(labelImage, dtype=torch.float32).unsqueeze(0))
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    
    def __len__(self):
        return len(self.inputs)
    
class Dataset_Predict(data.Dataset):
    def __init__(self, image):
        super(Dataset_Predict, self).__init__()
        self.inputs = []
        for j in range(image.shape[0] - 1):
            newImage = np.zeros((3, image.shape[1]), dtype=np.float32)
            newImage[0, :] = image[j,  :].copy()
            newImage[2, :] = image[j+1,  :].copy()
            self.inputs.append(torch.tensor(newImage, dtype=torch.float32).unsqueeze(0))

    def __getitem__(self, index):
        return self.inputs[index]

    def __len__(self):
        return len(self.inputs)
