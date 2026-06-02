import torch
from dataloader import make_dataset, NUM_CLASSES
from resnet34 import TinyResNet34
from classifier import train_one_epoch, evaluate
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

train_loader, val_loader = make_dataset(batch_size=128, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet34(NUM_CLASSES).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(200):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, NUM_CLASSES, use_mixup=True)
    va_loss, va_acc, cm = evaluate(model, val_loader, device, NUM_CLASSES)
    scheduler.step()
    print(f"epoch {epoch:3d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
