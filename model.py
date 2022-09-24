"""
A simple implementation of the 18-layer Residual Neural Network
as outlined in the original paper
by Kaiming He et al. https://arxiv.org/pdf/1512.03385.pdf
"""

import torch
import torch.nn as nn

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))


class ResBlock(nn.Module):
    relu = nn.ReLU()

    def __init__(self, in_ch, block_ch, downsample=None):
        super().__init__()
        self.downsample = downsample
        if downsample:
            # TODO: Use a zero padding shortcut
            #  to reduce memory/time complexity
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, block_ch, kernel_size=1,
                          stride=1, device=device),
                nn.BatchNorm2d(block_ch, device=device)
            )
        self.conv1 = nn.Conv2d(in_ch, block_ch,
                               kernel_size=3, padding=1, device=device)
        self.bn1 = nn.BatchNorm2d(block_ch, device=device)
        self.conv2 = nn.Conv2d(block_ch, block_ch,
                               kernel_size=3, padding=1, device=device)
        self.bn2 = nn.BatchNorm2d(block_ch, device=device)

    def forward(self, x):
        orig_x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            orig_x = self.downsample(orig_x)

        return self.relu(orig_x + x)


class ResStage(nn.Module):
    def __init__(self, in_ch, block_ch):
        super().__init__()
        self.block1 = ResBlock(in_ch, block_ch, downsample=True)
        self.block2 = ResBlock(block_ch, block_ch)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class MyResNet(nn.Module):
    relu = nn.ReLU()

    def __init__(self):
        super(MyResNet, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.stage1.to(device=device)

        self.conv2_x = ResStage(64, 64)
        self.conv3_x = ResStage(64, 128)
        self.conv4_x = ResStage(128, 256)
        self.conv5_x = ResStage(256, 512)

        # nn.AvgPool2d won't work because we need global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10, device=device)

        # TODO: Initialization of weights?

    def forward(self, x):
        x = self.stage1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.softmax(x, dim=1)
        return x


class Net(nn.Module):
    def __init__(self, img_sample, n_ch):
        super().__init__()
        self.img_h, self.img_w = img_sample.shape[-2:]
        assert self.img_h == self.img_w  # For now

        self.conv1 = nn.Conv2d(1, n_ch, kernel_size=3, padding=1, device=device)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(n_ch, int(n_ch / 2), kernel_size=3,
                               padding=1, device=device)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.sz = (self.conv2.out_channels *
                   int(self.img_h / self.pool1.kernel_size
                       / self.pool2.kernel_size)**2)
        self.fc1 = nn.Linear(self.sz, 32, device=device)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10, device=device)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, self.sz)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out
