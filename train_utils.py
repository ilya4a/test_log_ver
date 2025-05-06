import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Net
from log_test_ver import log_loaders, log_train_model

BATCH_SIZE = 256
LEARNING_RATE = 0.1
EPOCHS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@log_loaders
def get_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def evaluate_model(model, device, loader):
    model.eval()
    loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        if isinstance(loader, DataLoader):
            batches = loader
            total_samples = len(loader.dataset)
        else:
            batches = [loader] if not isinstance(loader, list) else loader
            total_samples = sum(batch[0].size(0) for batch in batches)

        for batch in batches:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, target = batch
            else:
                data, target = batch[0], batch[1]

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    if total_samples == 0:
        return float('nan'), 0.0

    loss /= total_samples
    accuracy = 100.0 * correct / total_samples
    return loss, accuracy


@log_train_model
def train_model(train_loader):
    """Инициализация и обучение модели"""
    model = Net().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)

    return model
