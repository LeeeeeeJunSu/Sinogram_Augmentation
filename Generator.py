import torch
import torch.nn as nn
import torch.nn.functional as F
import Layer

class Generator(nn.Module): # input is 1 x 3 x 512
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        # encoder x
        self.encoder_x_1 = Layer.conv(1, 32, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu', bn = False) # (B, 32, 3, 256)
        self.encoder_x_2 = Layer.conv(32, 64, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 64, 3, 128)
        self.encoder_x_3 = Layer.conv(64, 128, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 128, 3, 64)
        self.encoder_x_4 = Layer.conv(128, 256, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 256, 3, 32)
        self.encoder_x_5 = Layer.conv(256, 512, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 512, 3, 16)
        self.encoder_x_6 = Layer.conv(512, 512, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 512, 3, 8)
        self.encoder_x_7 = Layer.conv(512, 512, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu')  # (B, 512, 3, 4)
        self.encoder_x_8 = Layer.conv(512, 512, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 512, 3, 2)
        self.encoder_x_9 = Layer.conv(512, 512, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 512, 3, 1)

        # encoder y
        self.encoder_y_0 = Layer.conv(1, 32, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 32, 1, 512)
        self.encoder_y_1 = Layer.conv(32, 64, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 64, 1, 256)
        self.encoder_y_2 = Layer.conv(64, 128, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 128, 1, 128)
        self.encoder_y_3 = Layer.conv(128, 256, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 256, 1, 64)
        self.encoder_y_4 = Layer.conv(256, 512, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 512, 1, 32)
        self.encoder_y_5 = Layer.conv(512, 1024, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 1024, 1, 16)
        self.encoder_y_6 = Layer.conv(512, 1024, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 1024, 1, 8)
        self.encoder_y_7 = Layer.conv(512, 1024, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu')  # (B, 1024, 1, 4)
        self.encoder_y_8 = Layer.conv(512, 1024, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 1024, 1, 2)
        self.encoder_y_9 = Layer.conv(512, 1024, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu', bn=False) # (B, 1024, 1, 1)

        # encoder y + x
        self.encoder_yx_0 = Layer.conv(32, 64, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 64, 1, 256)
        self.encoder_yx_1 = Layer.conv(128, 128, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 128, 1, 128)
        self.encoder_yx_2 = Layer.conv(256, 256, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 256, 1, 64)
        self.encoder_yx_3 = Layer.conv(512, 512, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 512, 1, 32)
        self.encoder_yx_4 = Layer.conv(1024, 1024, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 1024, 1, 16)
        self.encoder_yx_5 = Layer.conv(2048, 1024, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 1024, 1, 8)
        self.encoder_yx_6 = Layer.conv(2048, 1024, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 1024, 1, 4)
        self.encoder_yx_7 = Layer.conv(2048, 1024, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu')  # (B, 1024, 1, 2)
        self.encoder_yx_8 = Layer.conv(2048, 1024, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu', bn=False) # (B, 1024, 1, 1)

        # decoder x
        self.decode_x_1 = Layer.deconv(2048, 1024, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 1024, 1, 2)
        self.decode_x_2 = Layer.deconv(2048, 1024, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 1024, 1, 4)
        self.decode_x_3 = Layer.deconv(2048, 1024, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu')  # (B, 1024, 1, 8)
        self.decode_x_4 = Layer.deconv(2048, 1024, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 1024, 1, 16)
        self.decode_x_5 = Layer.deconv(2048, 512, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 512, 1, 32)
        self.decode_x_6 = Layer.deconv(1024, 256, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 256, 1, 64)
        self.decode_x_7 = Layer.deconv(512, 128, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu')  # (B, 128, 1, 128)
        self.decode_x_8 = Layer.deconv(256, 64, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 64, 1, 256)
        self.decode_x_9 = Layer.deconv(128, 32, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 32, 1, 512)

        # decoder y
        self.decode_y_0 = Layer.deconv(2048, 1024, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 1024, 3, 1)
        self.decode_y_1 = Layer.deconv(1024, 512, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 512, 3, 2)
        self.decode_y_2 = Layer.deconv(1024, 512, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 512, 3, 4)
        self.decode_y_3 = Layer.deconv(1024, 512, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 512, 3, 8)
        self.decode_y_4 = Layer.deconv(1024, 512, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 512, 3, 16)
        self.decode_y_5 = Layer.deconv(512, 256, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 256, 3, 32)
        self.decode_y_6 = Layer.deconv(256, 128, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu')  # (B, 128, 3, 64)
        self.decode_y_7 = Layer.deconv(128, 64, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 64, 3, 128)
        self.decode_y_8 = Layer.deconv(64, 32, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 32, 3, 256)
        self.decode_y_9 = Layer.deconv(32, 1, k_size = (3, 1), stride = (1, 1), pad = (0, 0), activation='lrelu') # (B, 1, 3, 512)

        # decoder x + y
        self.decode_xy_0 = Layer.deconv(1536, 512, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 512, 3, 2)
        self.decode_xy_1 = Layer.deconv(1536, 512, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 512, 3, 4)
        self.decode_xy_2 = Layer.deconv(1536, 512, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 512, 3, 8)
        self.decode_xy_3 = Layer.deconv(1536, 512, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 512, 3, 16)
        self.decode_xy_4 = Layer.deconv(1536, 256, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 256, 3, 32)
        self.decode_xy_5 = Layer.deconv(768, 128, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu')  # (B, 128, 3, 64)
        self.decode_xy_6 = Layer.deconv(384, 64, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 64, 3, 128)
        self.decode_xy_7 = Layer.deconv(192, 32, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu')  # (B, 32, 3, 256)
        self.decode_xy_8 = Layer.deconv(96, 1, k_size = (1, 4), stride = (1, 2), pad = (0, 1), activation='lrelu') # (B, 1, 3, 512)
        self.decode_out = Layer.conv(2, 1, k_size = 1, stride = 1, pad = 0, activation='lrelu') # (B, 1024, 1, 1)

    # forward method
    def forward(self, input):
        original_size = (input.shape[2], input.shape[3])
        resizeInput = F.interpolate(input, size=(3, 512), mode='bilinear', align_corners=True)
        original_min = torch.min(resizeInput)
        original_max = torch.max(resizeInput)
        normalizedInput = (resizeInput - original_min) / (original_max - original_min)

        # Unet encoder
        ex1 = self.encoder_x_1(normalizedInput)
        ex2 = self.encoder_x_2(ex1)
        ex3 = self.encoder_x_3(ex2)
        ex4 = self.encoder_x_4(ex3)
        ex5 = self.encoder_x_5(ex4)
        ex6 = self.encoder_x_6(ex5)
        ex7 = self.encoder_x_7(ex6)
        ex8 = self.encoder_x_8(ex7)
        ex9 = self.encoder_x_9(ex8)

        ey0 = self.encoder_y_0(normalizedInput)
        ey1 = self.encoder_y_1(ex1)
        ey2 = self.encoder_y_2(ex2)
        ey3 = self.encoder_y_3(ex3)
        ey4 = self.encoder_y_4(ex4)
        ey5 = self.encoder_y_5(ex5)
        ey6 = self.encoder_y_6(ex6)
        ey7 = self.encoder_y_7(ex7)
        ey8 = self.encoder_y_8(ex8)
        ey9 = self.encoder_y_9(ex9)

        eyx0 = self.encoder_yx_0(ey0)
        eyx1 = self.encoder_yx_1(torch.cat((eyx0, ey1), 1))
        eyx2 = self.encoder_yx_2(torch.cat((eyx1, ey2), 1))
        eyx3 = self.encoder_yx_3(torch.cat((eyx2, ey3), 1))
        eyx4 = self.encoder_yx_4(torch.cat((eyx3, ey4), 1))
        eyx5 = self.encoder_yx_5(torch.cat((eyx4, ey5), 1))
        eyx6 = self.encoder_yx_6(torch.cat((eyx5, ey6), 1))
        eyx7 = self.encoder_yx_7(torch.cat((eyx6, ey7), 1))
        eyx8 = self.encoder_yx_8(torch.cat((eyx7, ey8), 1))
        
        # Unet decoder
        dx1 = self.decode_x_1(torch.cat((ey9, eyx8), 1))
        dx2 = self.decode_x_2(torch.cat((dx1, eyx7), 1))
        dx3 = self.decode_x_3(torch.cat((dx2, eyx6), 1))
        dx4 = self.decode_x_4(torch.cat((dx3, eyx5), 1))
        dx5 = self.decode_x_5(torch.cat((dx4, eyx4), 1))
        dx6 = self.decode_x_6(torch.cat((dx5, eyx3), 1))
        dx7 = self.decode_x_7(torch.cat((dx6, eyx2), 1))
        dx8 = self.decode_x_8(torch.cat((dx7, eyx1), 1))
        dx9 = self.decode_x_9(torch.cat((dx8, eyx0), 1))

        dy0 = self.decode_y_0(torch.cat((ey9, eyx8), 1))
        dy1 = self.decode_y_1(dx1)
        dy2 = self.decode_y_2(dx2)
        dy3 = self.decode_y_3(dx3)
        dy4 = self.decode_y_4(dx4)
        dy5 = self.decode_y_5(dx5)
        dy6 = self.decode_y_6(dx6)
        dy7 = self.decode_y_7(dx7)
        dy8 = self.decode_y_8(dx8)
        dy9 = self.decode_y_9(dx9)

        dxy0 = self.decode_xy_0(torch.cat((ex9, dy0), 1))
        dxy1 = self.decode_xy_1(torch.cat((ex8, dy1, dxy0), 1))
        dxy2 = self.decode_xy_2(torch.cat((ex7, dy2, dxy1), 1))
        dxy3 = self.decode_xy_3(torch.cat((ex6, dy3, dxy2), 1))
        dxy4 = self.decode_xy_4(torch.cat((ex5, dy4, dxy3), 1))
        dxy5 = self.decode_xy_5(torch.cat((ex4, dy5, dxy4), 1))
        dxy6 = self.decode_xy_6(torch.cat((ex3, dy6, dxy5), 1))
        dxy7 = self.decode_xy_7(torch.cat((ex2, dy7, dxy6), 1))
        dxy8 = self.decode_xy_8(torch.cat((ex1, dy8, dxy7), 1))
        output = self.decode_out(torch.cat((dy9, dxy8), 1))

        restoredOutput = output * (original_max - original_min) + original_min
        resizeOutput = F.interpolate(restoredOutput, size=original_size, mode='bilinear', align_corners=True)
        return resizeOutput
