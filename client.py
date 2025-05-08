import flwr as fl
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Load datasets
def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    return trainloader, testloader

# Define a simple CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define Flower Client
class FLClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.model = Net().to(DEVICE)
        self.trainloader, self.testloader = load_datasets(partition_id=int(cid))
        self.history = {
            'round': [],
            'train_loss': [],
            'test_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        self.current_round = 0

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.current_round += 1
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training phase
        self.model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in self.trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        self.history['round'].append(self.current_round)
        self.history['train_loss'].append(avg_train_loss)
        
        # Perform evaluation after training
        test_loss, accuracy, precision, recall, f1 = self.evaluate_model()
        
        # Store metrics in history
        self.history['test_loss'].append(test_loss)
        self.history['accuracy'].append(accuracy)
        self.history['precision'].append(precision)
        self.history['recall'].append(recall)
        self.history['f1'].append(f1)
        
        # Print current performance
        print(f"Client {self.cid} Round {self.current_round}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        
        return self.get_parameters(config), len(self.trainloader.dataset), {
            "train_loss": avg_train_loss,
            "test_loss": test_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_loss, accuracy, precision, recall, f1 = self.evaluate_model()
        
        return test_loss, len(self.testloader.dataset), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def evaluate_model(self):
        """Evaluate the model on the test dataset and return various metrics."""
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        
        test_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.testloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_test_loss = test_loss / len(self.testloader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Handle potential warning for labels with no predictions
        try:
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        except Exception as e:
            print(f"Warning in calculating metrics: {e}")
            precision = recall = f1 = 0.0
            
        return avg_test_loss, accuracy, precision, recall, f1
    
    def save_evaluation_results(self, filename=None):
        """Save evaluation history to a file."""
        if filename is None:
            filename = f"client_{self.cid}_evaluation_results.csv"
        
        import pandas as pd
        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)
        print(f"Evaluation results saved to {filename}")
        
        return filename


# Start the client and connect to the server
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("client_id", help="Client ID")
    parser.add_argument("--server", default="34.205.17.115:8080", help="Server address")
    parser.add_argument("--save", action="store_true", help="Save evaluation results after training")
    
    args = parser.parse_args()
    
    client = FLClient(args.client_id)
    fl.client.start_numpy_client(args.server, client=client)
    
    # Save results if requested
    if args.save:
        client.save_evaluation_results()