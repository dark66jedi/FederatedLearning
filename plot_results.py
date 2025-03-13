import os
import json
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Generate performance comparison graphs.")
parser.add_argument("--rerun", action="store_true", help="Re-run model training scripts even if results exist.")
args = parser.parse_args()

RESULTS_DIR = "results/data"
OUTPUT_DIR = "results/graphs"  # Saving all the images in the same directory

# Check if directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of model scripts and corresponding result files
models_info = {
    "cnn": "cnn_results.txt",
    "efficientnet": "efficientnet_results.txt",
    "mobilenet": "mobilenet_results.txt",
    "resnet": "resnet_results.txt",
}

# Run model scripts if necessary
for model, result_file in models_info.items():
    result_path = os.path.join(RESULTS_DIR, result_file)

    # Rerun if flag is set or file is missing
    if args.rerun or not os.path.exists(result_path):
        print(f"Running {model}.py ...")
        subprocess.run(["python3", f"{model}.py"])
    else:
        print(f"Skipping {model}.py (results exist).")


results = {}
models = []

for filename in os.listdir(RESULTS_DIR):
    if filename.endswith("_results.txt"):
        filepath = os.path.join(RESULTS_DIR, filename)
        with open(filepath, "r") as f:
            data = json.load(f)
            model_name = data["Model"]
            results[model_name] = data
            models.append(model_name)

if not results:
    print("No results found to plot.")
    exit()

# Extract metrics from results
train_accuracies = [results[m]["Train Accuracy"] for m in models]
train_losses = [results[m]["Train Loss"] for m in models]
test_accuracies = [results[m]["Test Accuracy"] for m in models]
test_losses = [results[m]["Test Loss"] for m in models]
training_times = [results[m]["Training Time (s)"] for m in models]

# Define positions for bars
x_pos = np.arange(len(models))

# Create subplots for each metric
# Plot Train Accuracy
fig_accuracy, ax_accuracy = plt.subplots(figsize=(10, 6))
fig_accuracy.suptitle("Train Accuracy Comparison", fontsize=14, fontweight="bold")
ax_accuracy.bar(x_pos, train_accuracies, color="skyblue")
ax_accuracy.set_ylabel("Train Accuracy")
ax_accuracy.set_xticks(x_pos)
ax_accuracy.set_xticklabels(models, ha="center")  # Centered labels, no rotation
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the title
fig_accuracy.savefig(os.path.join(OUTPUT_DIR, "train_accuracy_comparison.png"))
print("Train Accuracy plot saved as train_accuracy_comparison.png")

# Plot Train Loss
fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
fig_loss.suptitle("Train Loss Comparison", fontsize=14, fontweight="bold")
ax_loss.bar(x_pos, train_losses, color="skyblue")
ax_loss.set_ylabel("Train Loss")
ax_loss.set_xticks(x_pos)
ax_loss.set_xticklabels(models, ha="center")  # Centered labels, no rotation
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig_loss.savefig(os.path.join(OUTPUT_DIR, "train_loss_comparison.png"))
print("Train Loss plot saved as train_loss_comparison.png")

# Plot Test Accuracy
fig_test_accuracy, ax_test_accuracy = plt.subplots(figsize=(10, 6))
fig_test_accuracy.suptitle("Test Accuracy Comparison", fontsize=14, fontweight="bold")
ax_test_accuracy.bar(x_pos, test_accuracies, color="skyblue")
ax_test_accuracy.set_ylabel("Test Accuracy")
ax_test_accuracy.set_xticks(x_pos)
ax_test_accuracy.set_xticklabels(models, ha="center")  # Centered labels, no rotation
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the title
fig_test_accuracy.savefig(os.path.join(OUTPUT_DIR, "test_accuracy_comparison.png"))
print("Test Accuracy plot saved as test_accuracy_comparison.png")

# Plot Test Loss
fig_test_loss, ax_test_loss = plt.subplots(figsize=(10, 6))
fig_test_loss.suptitle("Test Loss Comparison", fontsize=14, fontweight="bold")
ax_test_loss.bar(x_pos, test_losses, color="skyblue")
ax_test_loss.set_ylabel("Test Loss")
ax_test_loss.set_xticks(x_pos)
ax_test_loss.set_xticklabels(models, ha="center")  # Centered labels, no rotation
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig_test_loss.savefig(os.path.join(OUTPUT_DIR, "test_loss_comparison.png"))
print("Test Loss plot saved as test_loss_comparison.png")

# Plot Training Time
fig_time, ax_time = plt.subplots(figsize=(10, 6))
fig_time.suptitle("Training Time Comparison", fontsize=14, fontweight="bold")
ax_time.bar(x_pos, training_times, color="skyblue")
ax_time.set_ylabel("Training Time (s)")
ax_time.set_xticks(x_pos)
ax_time.set_xticklabels(models, ha="center")  # Centered labels, no rotation
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig_time.savefig(os.path.join(OUTPUT_DIR, "training_time_comparison.png"))
print("Training Time plot saved as training_time_comparison.png")
