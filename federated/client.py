import flwr as fl
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

print(f"Using device: {DEVICE}")

# Load datasets
def load_datasets(partition_id: int):
    print(f"Loading dataset for partition {partition_id}...")
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
    partition = fds.load_partition(partition_id-1)
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
    
    print(f"Dataset loaded. Training samples: {len(partition_train_test['train'])}, Test samples: {len(partition_train_test['test'])}")
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
        print(f"Initializing client {cid}...")
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
        print(f"Client {cid} initialized successfully.")

    def get_parameters(self, config):
        print(f"Client {self.cid}: Getting parameters...")
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        print(f"Client {self.cid}: Setting parameters...")
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.current_round = config.get("round", self.current_round + 1)
        print(f"\nClient {self.cid} starting round {self.current_round} training...")
        self.set_parameters(parameters)

        # Get learning rate from config or use default
        lr = config.get("learning_rate", 0.001)
        epochs = config.get("epochs", 1)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Training phase
        self.model.train()
        train_loss = 0.0
        train_batches = 0
        
        for epoch in range(epochs):
            print(f"Client {self.cid} - Epoch {epoch+1}/{epochs}")
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_idx, batch in enumerate(self.trainloader):
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Client {self.cid} - Batch {batch_idx}/{len(self.trainloader)}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            print(f"Client {self.cid} - Epoch {epoch+1}/{epochs} complete. Avg Loss: {avg_epoch_loss:.4f}")
            
            train_loss += epoch_loss
            train_batches += epoch_batches
        
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
        print(f"Client {self.cid} Round {self.current_round} Results:")
        print(f"Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        
        return self.get_parameters(config), len(self.trainloader.dataset), {
            "train_loss": float(avg_train_loss),
            "test_loss": float(test_loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    def evaluate(self, parameters, config):
        print(f"Client {self.cid}: Starting evaluation...")
        self.set_parameters(parameters)
        test_loss, accuracy, precision, recall, f1 = self.evaluate_model()
        
        print(f"Client {self.cid} Evaluation Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        
        # FIXED: Return loss as 'loss' key instead of separate return value
        return float(test_loss), len(self.testloader.dataset), {
            "loss": float(test_loss),  # ADDED: Include loss in metrics dict
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    
    def evaluate_model(self):
        """Evaluate the model on the test dataset and return various metrics."""
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.testloader:
                # Properly extract images and labels from the batch dictionary
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Store for save_evaluation_results()
        self.all_preds = all_preds
        self.all_labels = all_labels

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        avg_loss = test_loss / len(self.testloader)

        return avg_loss, accuracy, precision, recall, f1
    
    def save_evaluation_results(self, output_dir=None):
        """Save evaluation history to a file and generate plots."""
        if output_dir is None:
            output_dir = f"./results/client_{self.cid}"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(self.history)
        metrics_file = os.path.join(output_dir, f"client_{self.cid}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Metrics saved to {metrics_file}")
        
        # Plot training and test loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['round'], self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(self.history['round'], self.history['test_loss'], 'r-', label='Test Loss')
        plt.title(f'Client {self.cid} - Training and Test Loss over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        loss_plot_file = os.path.join(output_dir, f"client_{self.cid}_loss.png")
        plt.savefig(loss_plot_file)
        plt.close()
        print(f"Loss plot saved to {loss_plot_file}")
        
        # Plot accuracy, precision, recall, and F1 score
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['round'], self.history['accuracy'], 'g-', label='Accuracy')
        plt.plot(self.history['round'], self.history['precision'], 'b-', label='Precision')
        plt.plot(self.history['round'], self.history['recall'], 'r-', label='Recall')
        plt.plot(self.history['round'], self.history['f1'], 'y-', label='F1 Score')
        plt.title(f'Client {self.cid} - Performance Metrics over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        metrics_plot_file = os.path.join(output_dir, f"client_{self.cid}_metrics.png")
        plt.savefig(metrics_plot_file)
        plt.close()
        print(f"Metrics plot saved to {metrics_plot_file}")
        
        # Save confusion matrix for the last round if available
        if len(self.all_labels) > 0 and len(self.all_preds) > 0:
            cm = confusion_matrix(self.all_labels, self.all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Client {self.cid} - Confusion Matrix (Last Round)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            cm_plot_file = os.path.join(output_dir, f"client_{self.cid}_confusion_matrix.png")
            plt.savefig(cm_plot_file)
            plt.close()
            print(f"Confusion matrix saved to {cm_plot_file}")
        
        # Save model
        model_path = os.path.join(output_dir, f"client_{self.cid}_model.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def save_confusion_matrix(self, output_dir=None):
        """Save confusion matrix of the current model state."""
        if output_dir is None:
            output_dir = f"./results/client_{self.cid}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate model and get predictions
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.testloader):
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Generate and save confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Client {self.cid} - Confusion Matrix (Round {self.current_round})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        cm_plot_file = os.path.join(output_dir, f"client_{self.cid}_confusion_matrix_round_{self.current_round}.png")
        plt.savefig(cm_plot_file)
        plt.close()
        print(f"Confusion matrix for round {self.current_round} saved to {cm_plot_file}")


def client_fn(cid):
    """Create a Flower client representing a single organization."""
    return FLClient(cid)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flower client example")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080",
                        help="Server address (default: 127.0.0.1:8080)")
    parser.add_argument("--cid", type=str, required=True,
                        help="Client ID (used for partitioning the dataset)")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory for storing logs")
    parser.add_argument("--result-dir", type=str, default="./results",
                        help="Directory for storing results")
    args = parser.parse_args()
    
    # Create client and result directory
    client = FLClient(cid=args.cid)
    result_dir = os.path.join(args.result_dir, f"client_{args.cid}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Configure logger
    log_dir = os.path.join(args.log_dir, f"client_{args.cid}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"client_{args.cid}.log")
    
    # Redirect stdout and stderr to log file
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up logging to file and console
    sys.stdout = open(log_file, "a")
    
    # Start Flower client
    print(f"\n{'=' * 50}")
    print(f"Starting Flower client {args.cid} with server {args.server_address}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 50}\n")
    
    try:
        # Start client
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client,
        )
        
        # Save final evaluation results
        client.save_evaluation_results(output_dir=result_dir)
        client.save_confusion_matrix(output_dir=result_dir)
        
        print(f"\n{'=' * 50}")
        print(f"Client {args.cid} finished successfully")
        print(f"Results saved to {result_dir}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 50}\n")
        
    except Exception as e:
        print(f"\n{'=' * 50}")
        print(f"Error in client {args.cid}: {e}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 50}\n")
        raise e


if __name__ == "__main__":
    main()