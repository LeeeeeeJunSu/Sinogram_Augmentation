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

class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        x = self.block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stage_1 = Dis_block(in_channels*2,64,normalize=False)
        self.stage_2 = Dis_block(64,128)
        self.stage_3 = Dis_block(128,256)
        self.stage_4 = Dis_block(256,512)
        self.patch = nn.Conv2d(512,1,3,padding=1) # 16x16 패치 생성
    def forward(self,a,b):
        x = torch.cat((a,b),1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x

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

class CustomEnginePix2Pix :
    def __init__(self, RootPath):
        self.RootPath = RootPath
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.GenertorPath = RootPath + "Generator\\Weight.pth"
        if not os.path.exists(RootPath + "Generator\\"):
            os.makedirs(RootPath + "Generator\\")
        self.Genertor = UNet()
        self.Genertor.to(self.device)
        if os.path.exists(self.GenertorPath):
            self.Genertor.load_state_dict(torch.load(self.GenertorPath))
        else:
            self.Genertor.apply(initialize_weights)
        self.DiscriminatorPath = RootPath + "Discriminator\\Weight.pth"
        if not os.path.exists(RootPath + "Discriminator\\"):
            os.makedirs(RootPath + "Discriminator\\")
        self.Discriminator = Discriminator()
        self.Discriminator.to(self.device)
        if os.path.exists(self.DiscriminatorPath):
            self.Discriminator.load_state_dict(torch.load(self.DiscriminatorPath))
        else:
            self.Discriminator.apply(initialize_weights)
    def Train(self):
        TrainDataset = CustomDataset(self.RootPath + "Train/")
        TrainLoader = DataLoader(TrainDataset, batch_size=32, shuffle=True)
        ValDataset = CustomDataset(self.RootPath + "Validation/")
        ValLoader = DataLoader(ValDataset, batch_size=32, shuffle=False)
        loss_func_gan = nn.BCELoss()
        loss_func_pix = nn.L1Loss()
        lambda_pixel = 100
        patch = (1,256//2**4,256//2**4)
        lr = 2e-4
        beta1 = 0.5
        beta2 = 0.999
        opt_dis = optim.Adam(self.Discriminator.parameters(),lr=lr,betas=(beta1,beta2))
        opt_gen = optim.Adam(self.Genertor.parameters(),lr=lr,betas=(beta1,beta2))
        self.Genertor.train()
        self.Discriminator.train()
        batch_count = 0
        num_epochs = 300
        start_time = time.time()
        log_dir = self.RootPath + 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=log_dir)
        for epoch in range(num_epochs):
            for a, b in TrainLoader:
                ba_si = a.size(0)
                real_a = a.to(self.device)
                real_b = b.to(self.device)
                self.Genertor.zero_grad()
                fake_b = self.Genertor(real_a)
                out_dis = self.Discriminator(fake_b, real_b)
                real_label = torch.ones_like(out_dis, requires_grad=False).to(self.device)
                fake_label = torch.zeros_like(out_dis, requires_grad=False).to(self.device)
                gen_loss = loss_func_gan(out_dis, real_label)
                pixel_loss = loss_func_pix(fake_b, real_b)
                g_loss = gen_loss + lambda_pixel * pixel_loss
                g_loss.backward()
                opt_gen.step()
                writer.add_scalar('Loss/train_g', g_loss.item(), batch_count)
                self.Discriminator.zero_grad()
                out_dis = self.Discriminator(real_b, real_a)
                real_loss = loss_func_gan(out_dis,real_label)
                out_dis = self.Discriminator(fake_b.detach(), real_a)
                fake_loss = loss_func_gan(out_dis,fake_label)
                d_loss = (real_loss + fake_loss) / 2.
                d_loss.backward()
                opt_dis.step()
                writer.add_scalar('Loss/train_d', d_loss.item(), batch_count)
                batch_count += 1
                if batch_count % 55 == 0:
                    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    print(f"Epoch: {epoch}, BatchCount: {batch_count}, Loss_G: {g_loss.item():.6f}, Loss_D {d_loss.item():.6f}, time: {(time.time() - start_time) / 60:.2f} min")
                    torch.save(self.Genertor.state_dict(), self.RootPath + f"Generator/Weight_epoch{epoch}_batch{batch_count}_loss{g_loss.item():.6f}_time{current_time}.pth")
                    torch.save(self.Discriminator.state_dict(), self.RootPath + f"Discriminator/Weight_epoch{epoch}_batch{batch_count}_loss{d_loss.item():.6f}_time{current_time}.pth")
            with torch.no_grad():
                val_g_loss = 0
                val_d_loss = 0
                val_count = 0
                self.Genertor.eval()
                self.Discriminator.eval()
                for a, b in ValLoader:
                    ba_si = a.size(0)
                    real_a = a.to(self.device)
                    real_b = b.to(self.device)
                    fake_b = self.Genertor(real_a)
                    out_dis = self.Discriminator(fake_b, real_b)
                    real_label = torch.ones_like(out_dis, requires_grad=False).to(self.device)
                    fake_label = torch.zeros_like(out_dis, requires_grad=False).to(self.device)
                    gen_loss = loss_func_gan(out_dis, real_label)
                    pixel_loss = loss_func_pix(fake_b, real_b)
                    g_loss = gen_loss + lambda_pixel * pixel_loss
                    val_g_loss += g_loss.item()
                    out_dis = self.Discriminator(real_b, real_a)
                    real_loss = loss_func_gan(out_dis, real_label)
                    out_dis = self.Discriminator(fake_b, real_a)
                    fake_loss = loss_func_gan(out_dis, fake_label)
                    d_loss = (real_loss + fake_loss) / 2.
                    val_d_loss += d_loss.item()
                    val_count += 1
                writer.add_scalar('Loss/val_g', val_g_loss / val_count, batch_count)
                writer.add_scalar('Loss/val_d', val_d_loss / val_count, batch_count)
                print(f"Epoch: {epoch}, Val Loss_G: {val_g_loss / val_count:.6f}, Val Loss_D: {val_d_loss / val_count:.6f}, time: {(time.time() - start_time) / 60:.2f} min")
        torch.save(self.Genertor.state_dict(), self.GenertorPath)
        torch.save(self.Discriminator.state_dict(), self.DiscriminatorPath)
        writer.close()
    def CreateSinogram(self, InputPath, OutputPath):
        self.Genertor.eval()
        TestDataset = CustomDatasetForResult(InputPath)
        TestLoader = DataLoader(TestDataset, batch_size=1, shuffle=False)
        output = []
        with torch.no_grad():
            for a, b in TestLoader:
                real_a = a.to(self.device)
                real_b = b.to(self.device)
                fake_a = self.Genertor(real_a)
                fake_b = fake_a * real_b
                output.append(fake_b[0][0].cpu().numpy())
        output = np.array(output)
        output.tofile(OutputPath)
