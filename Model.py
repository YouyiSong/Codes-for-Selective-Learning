import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, num):
        super(UNet, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.Dropout(0.1),
                                   nn.InstanceNorm2d(64),
                                   nn.LeakyReLU(0.01, inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.Dropout(0.1),
                                   nn.InstanceNorm2d(64),
                                   nn.LeakyReLU(0.01, inplace=True)
                                   )

        self.down2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
                                   nn.Dropout(0.1),
                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.Dropout(0.1),
                                   nn.InstanceNorm2d(128),
                                   nn.LeakyReLU(0.01, inplace=True),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.Dropout(0.1),
                                   nn.InstanceNorm2d(128),
                                   nn.LeakyReLU(0.01, inplace=True)
                                   )

        self.down3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
                                   nn.Dropout(0.1),
                                   nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.Dropout(0.1),
                                   nn.InstanceNorm2d(256),
                                   nn.LeakyReLU(0.01, inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.Dropout(0.1),
                                   nn.InstanceNorm2d(256),
                                   nn.LeakyReLU(0.01, inplace=True)
                                   )

        self.down4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                                   nn.Dropout(0.1),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.Dropout(0.1),
                                   nn.InstanceNorm2d(512),
                                   nn.LeakyReLU(0.01, inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.Dropout(0.1),
                                   nn.InstanceNorm2d(512),
                                   nn.LeakyReLU(0.01, inplace=True)
                                   )

        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=True),
                                    nn.Dropout(0.1),
                                    nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.Dropout(0.1),
                                    nn.InstanceNorm2d(1024),
                                    nn.LeakyReLU(0.01, inplace=True),
                                    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.Dropout(0.1),
                                    nn.InstanceNorm2d(1024),
                                    nn.LeakyReLU(0.01, inplace=True),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=True),
                                    nn.Dropout(0.1)
                                    )

        self.up1 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.Dropout(0.1),
                                 nn.InstanceNorm2d(512),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.Dropout(0.2),
                                 nn.InstanceNorm2d(512),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=True),
                                 nn.Dropout(0.1)
                                 )

        self.up2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.Dropout(0.1),
                                 nn.InstanceNorm2d(256),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.Dropout(0.1),
                                 nn.InstanceNorm2d(256),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=True),
                                 nn.Dropout(0.1)
                                 )

        self.up3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.Dropout(0.1),
                                 nn.InstanceNorm2d(128),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.Dropout(0.1),
                                 nn.InstanceNorm2d(128),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=True),
                                 nn.Dropout(0.1)
                                 )

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.Dropout(0.1),
                                 nn.InstanceNorm2d(64),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.Dropout(0.1),
                                 nn.InstanceNorm2d(64),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv2d(64, num, kernel_size=1, bias=True),
                                 nn.Dropout(0.1),
                                 nn.Softmax(dim=1)
                                 )

    def forward(self, img):
        x1 = self.down1(img)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.center(x4)
        x = self.up1(torch.cat([x, x4], dim=1))
        x = self.up2(torch.cat([x, x3], dim=1))
        x = self.up3(torch.cat([x, x2], dim=1))
        x = self.up4(torch.cat([x, x1], dim=1))
        return x