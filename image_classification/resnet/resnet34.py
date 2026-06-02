import torch
import torch.nn as nn
from TinyResNet import BasicBlock

class TinyResNet34(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                )

        self.layer1 = self._make_group(64, 64, num_blocks=3, stride=1)
        self.layer2 = self._make_group(64, 128, num_blocks=4, stride=2)
        self.layer3 = self._make_group(128, 256, num_blocks=6, stride=2)
        self.layer4 = self._make_group(256, 512, num_blocks=3, stride=2)

        self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 100),
                )


    def _make_group(self, in_c, out_c, num_blocks, stride):
        blocks = [BasicBlock(in_c, out_c, stride=stride)]
        for _ in range(num_blocks-1):
            blocks.append(BasicBlock(out_c, out_c, stride=1))
        return nn.Sequential(*blocks)


    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return(self.head(x))

