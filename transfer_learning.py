from torchvision.models import resnet18, ResNet18_Weights

r18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
r18.eval()
print(f"ResNet-18 params: {sum(p.numel() for p in r18.parameters())}")
#print(r18)
for p in r18.parameters:
    p.requires_grad = False
r18.fc = nn.Linear(r18.fc.in_features, 10)
