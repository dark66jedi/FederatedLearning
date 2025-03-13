import time
import json
import os
import torch
import torchvision.models as models
from collections import OrderedDict
from typing import List
from flwr.client import NumPyClient
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 10
BATCH_SIZE = 32
RESULTS_DIR = "results/data"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader

class ResNet(nn.Module):
    def __init__(self) -> None:
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def train(net, trainloader, epochs: int):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        
    return epoch_loss / len(trainloader.dataset), correct / total

def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(testloader.dataset), correct / total

def save_results(results, filename="resnet_results.txt"):
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

def main():
    start_time = time.time()
    net = ResNet().to(DEVICE)
    trainloader, valloader, testloader = load_datasets(partition_id=0)
    
    train_loss, train_acc = train(net, trainloader, epochs=5)
    test_loss, test_acc = test(net, testloader)
    
    elapsed_time = time.time() - start_time
    results = {
        "Model": "ResNet",
        "Train Loss": train_loss,
        "Train Accuracy": train_acc,
        "Test Loss": test_loss,
        "Test Accuracy": test_acc,
        "Training Time (s)": elapsed_time
    }
    save_results(results)
    print(f"Results saved to {RESULTS_DIR}/resnet_results.txt")

if __name__ == "__main__":
    main()
