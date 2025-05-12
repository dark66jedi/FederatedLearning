import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 10  # Total number of epochs for classical training
NUM_CLIENTS = 10  # Number of clients/partitions in federated dataset

print(f"Using device: {DEVICE}")

# Load the full centralized dataset from all partitions
def load_centralized_dataset():
    print("Loading centralized dataset...")
    
    train_datasets = []
    test_datasets = []
    
    # Use the same dataset and partitioning as in federated learning
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    
    for i in range(NUM_CLIENTS):
        print(f"Loading partition {i}...")
        partition = fds.load_partition(i)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        
        pytorch_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        
        def apply_transforms(batch):
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch
        
        partition_train_test = partition_train_test.with_transform(apply_transforms)
        
        # Add to our collections
        train_datasets.append(partition_train_test["train"])
        test_datasets.append(partition_train_test["test"])
    
    # Combine all partitions into one centralized dataset
    combined_train = ConcatDataset(train_datasets)
    combined_test = ConcatDataset(test_datasets)
    
    # Create data loaders
    trainloader = DataLoader(combined_train, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(combined_test, batch_size=BATCH_SIZE)
    
    print(f"Combined training set size: {len(combined_train)}")
    print(f"Combined test set size: {len(combined_test)}")
    
    return trainloader, testloader

# Define the same CNN model as used in federated learning
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

class ClassicalTrainer:
    def __init__(self):
        self.model = Net().to(DEVICE)
        self.trainloader, self.testloader = load_centralized_dataset()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.history = {
            'epoch': [],
            'train_loss': [],
            'test_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'training_time': []
        }
    
    def train_one_epoch(self, epoch):
        """Train the model for one epoch and return the average loss."""
        self.model.train()
        running_loss = 0.0
        
        start_time = time.time()
        for i, batch in enumerate(self.trainloader):
            # Get the inputs and labels
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.4f}")
                running_loss = 0.0
        
        training_time = time.time() - start_time
        return running_loss / len(self.trainloader), training_time
    
    def evaluate_model(self):
        """Evaluate the model on the test dataset."""
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.testloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_test_loss = test_loss / len(self.testloader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_test_loss, accuracy, precision, recall, f1
    
    def train(self):
        """Train the model for a specified number of epochs."""
        print(f"Starting classical training for {NUM_EPOCHS} epochs...")
        
        for epoch in range(NUM_EPOCHS):
            # Train one epoch
            train_loss, training_time = self.train_one_epoch(epoch)
            
            # Evaluate on test data
            test_loss, accuracy, precision, recall, f1 = self.evaluate_model()
            
            # Store metrics in history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['accuracy'].append(accuracy)
            self.history['precision'].append(precision)
            self.history['recall'].append(recall)
            self.history['f1'].append(f1)
            self.history['training_time'].append(training_time)
            
            # Print current performance
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}:")
            print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")
            print(f"Training time: {training_time:.2f} seconds")
            print("-" * 60)
        
        print("Training completed!")
    
    def save_model(self, path="classical_model.pth"):
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="classical_model.pth"):
        """Load a trained model."""
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    
    def save_results(self, filename="classical_training_results.csv"):
        """Save training history to a CSV file."""
        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def plot_metrics(self, save_dir=None):
        """Plot training and evaluation metrics as separate image files."""
        # Create the save directory if it doesn't exist and a path is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Plot loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epoch'], self.history['train_loss'], 'b-', label='Train Loss')
        plt.plot(self.history['epoch'], self.history['test_loss'], 'r-', label='Test Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_dir:
            loss_path = os.path.join(save_dir, "loss.png")
            plt.savefig(loss_path)
            print(f"Loss plot saved to {loss_path}")
        plt.show()
        plt.close()
        
        # Plot accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epoch'], self.history['accuracy'], 'g-', label='Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_dir:
            accuracy_path = os.path.join(save_dir, "accuracy.png")
            plt.savefig(accuracy_path)
            print(f"Accuracy plot saved to {accuracy_path}")
        plt.show()
        plt.close()
        
        # Plot precision and recall
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epoch'], self.history['precision'], 'm-', label='Precision')
        plt.plot(self.history['epoch'], self.history['recall'], 'c-', label='Recall')
        plt.title('Precision and Recall over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_dir:
            pr_path = os.path.join(save_dir, "precision_recall.png")
            plt.savefig(pr_path)
            print(f"Precision and Recall plot saved to {pr_path}")
        plt.show()
        plt.close()
        
        # Plot F1 score
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epoch'], self.history['f1'], 'orange', label='F1 Score')
        plt.title('F1 Score over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_dir:
            f1_path = os.path.join(save_dir, "f1_score.png")
            plt.savefig(f1_path)
            print(f"F1 Score plot saved to {f1_path}")
        plt.show()
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classical Machine Learning Training")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs for training")
    parser.add_argument("--save_model", action="store_true", help="Save the trained model")
    parser.add_argument("--save_results", action="store_true", help="Save training results")
    parser.add_argument("--plot", action="store_true", help="Plot training metrics")
    parser.add_argument("--output_dir", default="./results", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Update number of epochs if specified
    if args.epochs != NUM_EPOCHS:
        NUM_EPOCHS = args.epochs
    
    # Create output directory if needed
    if args.save_model or args.save_results or args.plot:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer and train model
    trainer = ClassicalTrainer()
    trainer.train()
    
    # Save model if requested
    if args.save_model:
        model_path = os.path.join(args.output_dir, "classical_model.pth")
        trainer.save_model(model_path)
    
    # Save results if requested
    if args.save_results:
        results_path = os.path.join(args.output_dir, "classical_training_results.csv")
        trainer.save_results(results_path)
    
    # Plot and save metrics if requested
    if args.plot:
        # Create a plots subdirectory
        plots_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        trainer.plot_metrics(plots_dir)