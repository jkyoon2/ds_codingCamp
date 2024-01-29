import torch
from torch import nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class SimpleVGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = VGGBlock(3, 64, 128)
        self.pool = nn.MaxPool2d(2, 2)  # 고정된 출력 크기를 갖는 풀링 레이어
        self.fc = nn.Linear(128 * 112 * 112, num_classes)  # 풀링 후의 크기에 맞게 입력 크기 조정

    def forward(self, x):
        x = self.block1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x
