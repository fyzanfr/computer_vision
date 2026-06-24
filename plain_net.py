import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if in_c != out_c or stride != 1:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_c),
                    )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)



class PlainBlock(nn.Module):
     def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out)


def make_network(block_class, num_classes=10):
    stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            )

    groups = [
            (32, 64, 6),
            (64, 128, 8),
            (128, 256, 12),
            (256, 512, 6),
            ]

    layers = []
    for in_c, out_c, block_num in groups:
        blocks = [block_class(in_c, out_c, stride=2)]
        for _ in range(block_num - 1):
            blocks.append(block_class(in_c, out_c, stride=1))
        layers.append(nn.Sequential(*blocks))
    head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
            )
    return nn.Sequential(stem, *layers, head)

net = TinyResNet()
x = torch.randn(1, 3, 32, 32)
print(f"output: {net(x).shape}")
print(f"params: {sum(p.numel() for p in net.parameters())}")





#She brought upon me so much heaviness,
#With the affright that from her aspect came,
#That I the hope relinquished of the height.
#And as he is who willingly acquires,
#And the time comes that causes him to lose,
#Who weeps in all his thoughts and is despondent,
#E’en such made me that beast withouten peace,
#Which, coming on against me by degrees
#Thrust me back thither where the sun is silent 16
#While I was rushing downward to the lowland,
#Before mine eyes did one present himself,
#Who seemed from long-continued silence hoarse
