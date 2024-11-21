import os
import time
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import datetime
from os import listdir
from os.path import join
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image

def initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.down = nn.Sequential(*layers)
    def forward(self, x):
        x = self.down(x)
        return x

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False), nn.InstanceNorm2d(out_channels), nn.LeakyReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.up = nn.Sequential(*layers)
    def forward(self, x, skip):
        x = self.up(x)
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, skip), 1)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)                 
        self.down3 = UNetDown(128, 256)               
        self.down4 = UNetDown(256, 512, normalize=False, dropout=0.2)
        self.up1 = UNetUp(512, 256, dropout=0.2)
        self.up2 = UNetUp(512, 128, dropout=0.2)
        self.up3 = UNetUp(256, 64, dropout=0.2)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3)
        return u4

class CustomDataset(Dataset):
    def __init__(self, RootPath):
        super().__init__()
        self.RootPath = RootPath
        self.img_filenames = [f for f in listdir(RootPath) if not f.endswith("_Label.raw")]
    def __getitem__(self, index):
        a = np.fromfile(join(self.RootPath, self.img_filenames[index]), dtype=np.float32).reshape([1,720,512])
        b = np.fromfile(join(self.RootPath, self.img_filenames[index].replace("_Input.raw", "_Label.raw")), dtype=np.float32).reshape([1,720,512])
        max_b = np.max(b)
        a = a / max_b
        b = b / max_b
        return a,b
    def __len__(self):
        return len(self.img_filenames)

class CustomDatasetForResult(Dataset):
    def __init__(self, ImagePath):
        super().__init__()
        self.ImagePath = ImagePath
        self.FullImage = np.fromfile(ImagePath, dtype=np.float32).reshape([512,720,512])
        self.Slice = []
        for i in range(512):
            self.Slice.append(self.FullImage[i, :, :])

    def __getitem__(self, index):
        a = self.Slice[index]
        max_a = np.max(a)
        a = a / max_a
        a = torch.tensor(a).reshape([1,720,512]).float()
        return a,max_a

    def __len__(self):
        return len(self.Slice)

class CustomEngineUNet:
    def __init__(self, RootPath):
        self.RootPath = RootPath
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ModelPath = RootPath + "UNet/Weight.pth"
        if not os.path.exists(RootPath + "UNet/"):
            os.makedirs(RootPath + "UNet/")
        self.Model = UNet()
        self.Model.to(self.device)
        if os.path.exists(self.ModelPath):
            self.Model.load_state_dict(torch.load(self.ModelPath))
        else:
            self.Model.apply(initialize_weights)

    def Train(self):
        TrainDataset = CustomDataset(self.RootPath + "Train/")
        TrainLoader = DataLoader(TrainDataset, batch_size=32, shuffle=True)
        ValDataset = CustomDataset(self.RootPath + "Validation/")
        ValLoader = DataLoader(ValDataset, batch_size=32, shuffle=False)
        loss_func = nn.MSELoss()
        lr = 2e-4
        beta1 = 0.5
        beta2 = 0.999
        opt = optim.Adam(self.Model.parameters(), lr=lr, betas=(beta1, beta2))
        self.Model.train()
        batch_count = 0
        num_epochs = 300
        start_time = time.time()
        log_dir = self.RootPath + 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=log_dir)
        for epoch in range(num_epochs):
            self.Model.train()
            for a, b in TrainLoader:
                real_a = a.to(self.device)
                real_b = b.to(self.device)
                self.Model.zero_grad()
                fake_b = self.Model(real_a)
                loss = loss_func(fake_b, real_b)
                loss.backward()
                opt.step()
                writer.add_scalar('Loss/train', loss.item(), batch_count)
                batch_count += 1
                if batch_count % 55 == 0:
                    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    print(f"Epoch: {epoch}, BatchCount: {batch_count}, Loss: {loss.item():.6f}, time: {(time.time() - start_time) / 60:.2f} min")
                    torch.save(self.Model.state_dict(), self.RootPath + f"UNet/Weight_epoch{epoch}_batch{batch_count}_loss{loss.item():.6f}_time{current_time}.pth")
            with torch.no_grad():
                val_loss = 0
                val_count = 0
                self.Model.eval()
                for a, b in ValLoader:
                    real_a = a.to(self.device)
                    real_b = b.to(self.device)
                    fake_b = self.Model(real_a)
                    loss = loss_func(fake_b, real_b)
                    val_loss += loss.item()
                    val_count += 1
                writer.add_scalar('Loss/val', val_loss / val_count, batch_count)
                print(f"Epoch: {epoch}, Val Loss: {val_loss / val_count:.6f}, time: {(time.time() - start_time) / 60:.2f} min")
        torch.save(self.Model.state_dict(), self.ModelPath)
        writer.close()

    def Predict(self):
        self.Model.eval()
        TestDataset = CustomDataset(self.RootPath + "Test/")
        TestLoader = DataLoader(TestDataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for a, b in TestLoader:
                print(a.shape)
                real_a = a.to(self.device)
                real_b = b.to(self.device)
                fake_b = self.Model(real_a)
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(real_a[0][0].cpu().numpy(), cmap='gray')
                plt.title('Input Image')
                plt.subplot(1, 3, 2)
                plt.imshow(real_b[0][0].cpu().numpy(), cmap='gray')
                plt.title('Label Image')
                plt.subplot(1, 3, 3)
                plt.imshow(fake_b[0][0].cpu().numpy(), cmap='gray')
                plt.title('Generated Image')
                plt.show()

    def CreateSinogram(self, InputPath, OutputPath):
        self.Model.eval()
        TestDataset = CustomDatasetForResult(InputPath)
        TestLoader = DataLoader(TestDataset, batch_size=1, shuffle=False)
        output = []
        with torch.no_grad():
            for a, b in TestLoader:
                real_a = a.to(self.device)
                real_b = b.to(self.device)
                fake_a = self.Model(real_a)
                fake_b = fake_a * real_b
                output.append(fake_b[0][0].cpu().numpy())
        output = np.array(output)
        output.tofile(OutputPath)


