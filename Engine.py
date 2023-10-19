import torch
import torch.nn as nn
import torch.optim as optim
import os
from tifffile import imread
import numpy as np
from torch.utils.data import DataLoader
import Generator
import Discriminator
import CustomDataset

class engine():
    def __init__(self):
        self.G = Generator.Generator()
        self.G.to('cuda')
        self.D = Discriminator.Discriminator()
        self.D.to('cuda')

    # 모델을 로드한다. 이때 모델의 이름은 Generator.pt, Discriminator.pt로 고정한다.
    # ModelFolder: 모델이 저장된 폴더    
    def loadModel(self, ModelFolder):
        self.G.load_state_dict(torch.load(ModelFolder + "Generator.pt"))
        self.D.load_state_dict(torch.load(ModelFolder + "Discriminator.pt"))

    # 모델을 저장한다. 이때 모델의 이름은 Generator.pt, Discriminator.pt로 고정한다.
    # ModelFolder: 모델을 저장할 폴더
    def saveModel(self, ModelFolder):
        torch.save(self.G.state_dict(), ModelFolder + "Generator.pt")
        torch.save(self.D.state_dict(), ModelFolder + "Discriminator.pt")

    # 모델을 학습한다.
    # trainFolder: 학습에 사용할 이미지가 저장된 폴더
    # trainFolder 내부의 이미지는 모두 16bit tiff 파일이다.
    # trainFolder 내부의 이미지를 로드 하여 y0 ~ y2, y1~y3, y2~y4... 의 형태로 이미지를 파싱한다.
    # 파싱한 이미지의 중간 열을 0으로 처리한 이미지를 인풋으로 사용하고, 파싱한 이미지를 라벨로 사용한다.
    def train(self, trainFolder, epochs = 10, batchSize = 720, learningRate = 0.0002, betas = (0.5,0.999), debugStep = 10, delete_zero = True):
        self.G.train()
        self.D.train()

        criterionL1 = nn.L1Loss().cuda()
        criterionMSE = nn.MSELoss().cuda()
        g_optimizer = optim.Adam(self.parameters(), lr=learningRate, betas=betas)
        d_optimizer = optim.Adam(self.parameters(), lr=learningRate, betas=betas)

        train_dataset = CustomDataset.Dataset_Train(trainFolder, delete_zero)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
        for epoch in range(1, epochs):
            for i, (real_a, real_b) in enumerate(train_loader, 1):
                for i, (real_a, real_b) in enumerate(train_loader, 1):
                    # forward
                    real_a, real_b = real_a.cuda(), real_b.cuda()
                    real_label = torch.ones(1).cuda()
                    fake_label = torch.zeros(1).cuda()
                    fake_b = self.G(real_a) # G가 생성한 fake Segmentation mask
                    #============= Train the discriminator =============#
                    # train with fake
                    fake_ab = torch.cat((real_a, fake_b), 1)
                    pred_fake = self.D.forward(fake_ab.detach())
                    loss_d_fake = criterionMSE(pred_fake, fake_label)
                    # train with real
                    real_ab = torch.cat((real_a, real_b), 1)
                    pred_real = self.D.forward(real_ab)
                    loss_d_real = criterionMSE(pred_real, real_label)
                    # Combined D loss
                    loss_d = (loss_d_fake + loss_d_real) * 0.5
                    # Backprop + Optimize
                    self.D.zero_grad()
                    loss_d.backward()
                    d_optimizer.step()
                    #=============== Train the generator ===============#
                    # First, G(A) should fake the discriminator
                    fake_ab = torch.cat((real_a, fake_b), 1)
                    pred_fake = self.D.forward(fake_ab)
                    loss_g_gan = criterionMSE(pred_fake, real_label)
                    # Second, G(A) = B
                    loss_g_l1 = criterionL1(fake_b, real_b) * 10
                    loss_g = loss_g_gan + loss_g_l1
                    # Backprop + Optimize
                    self.G.zero_grad()
                    self.D.zero_grad()
                    loss_g.backward()
                    g_optimizer.step()
                    if i % debugStep == 0:
                        print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f' % (epoch, epochs, i, len(train_loader), loss_d.item(), loss_g.item()))
    
    # 모델을 통해 이미지를 생성한다.
    # imageFolder: Sinogram Augmentation에 사용될 원본 이미지가 저장된 폴더
    # imageFolder 내부의 이미지는 모두 16bit tiff 파일이다.
    # imageFolder 내부의 이미지를 로드 하여 y0 ~ y1, y1~y2, y2~y3... 의 형태로 이미지를 파싱한다.
    # 파싱한 이미지에 중간에 열을 추가하여 인풋으로 사용한다.
    # 파싱한 이미지의 0번 열과 Generator가 생성한 이미지의 1번 열을 사용하여 원본 이미지 Height * 2 - 1의 이미지를 생성한다.
    # 원본 이미지의 파일 이름, 생성된 이미지를 튜플로 묶어서 리스트에 저장하여 리턴한다.
    def predict(self, imageFolder):
        self.G.eval()
        resultList = []
        path = os.path.join(imageFolder) 
        imageNames = sorted([f for f in os.listdir(path) if f.endswith('.tif')])
        for i in imageNames:
            filename = os.path.join(path, i)
            oneImage = imread(filename)
            newImage = np.zeros(((oneImage.shape[0] * 2) - 1, oneImage.shape[1]), dtype=np.float32)
            test_dataset = CustomDataset.Dataset_Predict(oneImage)
            test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
            for j, (input) in enumerate(test_loader, 1):
                input = input.cuda()
                result = self.G(input)
                input_np = input.detach().squeeze().cpu().numpy()
                result_np = result.detach().squeeze().cpu().numpy()
                newImage[((j - 1) * 2), :] = input_np[0, :]
                newImage[((j - 1) * 2) + 1, :] = result_np[1, :]
            resultList.append((i, newImage))
        return resultList
