""" Full assembly of the parts to form the complete network """

from .unet_parts import *


import torch
import torch.nn as nn
import torch.nn.functional as F
from Networks.DP_CoNet import EPEDLayer
class UNet_EPED(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,use_softmax= False):
        super(UNet_EPED, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))


        self.fsa64=EPEDLayer(64)
        self.fsa128=EPEDLayer(128)
        self.fsa256=EPEDLayer(256)
        self.fsa512=EPEDLayer(512)
        self.fsa1024=EPEDLayer(1024)    
        # self.use_softmax = use_softmax
        # if self.use_softmax==True:
        #     self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2=self.fsa128(x2)
        x3 = self.down2(x2)
        x3=self.fsa256(x3)
        x4 = self.down3(x3)
        x4=self.fsa512(x4)
        x5 = self.down4(x4)
        x5=self.fsa1024(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # if self.use_softmax==True:
        #     logits = self.softmax(logits)

        return logits



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x




# Attunet

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

