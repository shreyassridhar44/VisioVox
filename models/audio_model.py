# models/audio_model.py  -  VisioVox v4
# AudioUNet with Dropout2d(0.2) in decoder
# Dropout prevents memorisation of training mixtures (overfitting fix)

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioUNet(nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()
        self.enc1       = self._blk(1,   32)
        self.enc2       = self._blk(32,  64)
        self.enc3       = self._blk(64,  128)
        self.enc4       = self._blk(128, 256)
        self.bottleneck = self._blk(256, 512)
        self.pool       = nn.MaxPool2d(2)
        self.drop       = nn.Dropout2d(p=dropout_p)

        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._blk(512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._blk(256, 128)
        self.up2  = nn.ConvTranspose2d(128,  64, 2, stride=2)
        self.dec2 = self._blk(128,  64)
        self.up1  = nn.ConvTranspose2d( 64,  32, 2, stride=2)
        self.dec1 = self._blk( 64,  32)
        self.out  = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def _blk(self, ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )

    @staticmethod
    def _pad(x, ref):
        dh = ref.size(2) - x.size(2)
        dw = ref.size(3) - x.size(3)
        return F.pad(x, [dw//2, dw - dw//2, dh//2, dh - dh//2])

    def encode(self, x):
        e1 = self.enc1(x);  p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2); p3 = self.pool(e3)
        e4 = self.enc4(p3); p4 = self.pool(e4)
        b  = self.bottleneck(p4)
        return b, [e1, e2, e3, e4]

    def decode(self, x, skips):
        e1, e2, e3, e4 = skips
        x = self.drop(x)
        x = self.dec4(torch.cat([self._pad(self.up4(x), e4), e4], 1))
        x = self.drop(x)
        x = self.dec3(torch.cat([self._pad(self.up3(x), e3), e3], 1))
        x = self.drop(x)
        x = self.dec2(torch.cat([self._pad(self.up2(x), e2), e2], 1))
        x = self.dec1(torch.cat([self._pad(self.up1(x), e1), e1], 1))
        return self.out(x)


# Aliases for backward compatibility
AudioEncoder = AudioUNet
AudioDecoder = AudioUNet