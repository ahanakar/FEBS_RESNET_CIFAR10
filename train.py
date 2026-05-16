import torch 
import torch.nn as nn
import torch.optim as optim 
from model import ResNet18 
from dataset import get_dataloaders

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

Batch_Size = 128
Epochs = 164
learning_rate = 0.1

train_loader, test_loader = get_dataloaders(batch_size = Batch_Size)
model = ResNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = 1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [82, 123], gamma = 0.1)

train_losses, train_accs = [], []
val_losses, val_accs = [], []

def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    print(f"Epoch: {epoch} | Training loss: {epoch_loss:.4f} | Training accuracy: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc
    
def validate(epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total 
    print(f"Epoch: {epoch} | Validation loss: {epoch_loss:.4f} | Validation accuracy: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    for epoch in range(1, Epochs + 1):
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = validate(epoch)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

    torch.save(model.state_dict(), 'resnet18_cifar10.pth')
    print("training complete, weights saved")