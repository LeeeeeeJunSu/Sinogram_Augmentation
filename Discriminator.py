import torch.nn as nn
import Layer

class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Layer.conv(2, 64, 4, pad = (2,1), bn=False, activation='lrelu')
        self.conv2 = Layer.conv(64, 128, 4, pad = (2,1), activation='lrelu')
        self.conv3 = Layer.conv(128, 256, 4, pad = (2,1), activation='lrelu')
        self.conv4 = Layer.conv(256, 512, 4, 1, pad = (2,1), activation='lrelu')
        self.conv5 = Layer.conv(512, 1, 4, 1, pad = (2,1), activation='none')

    # forward method
    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return out