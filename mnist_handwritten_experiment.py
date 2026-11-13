"""MNIST handwritten digit recognition experiment (Example 4.1).
Run once: python mnist_handwritten_experiment.py
"""
from pathlib import Path
import csv
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -------------------- Settings --------------------
DATA_ROOT = Path(r"D:\data")
OUTPUT_DIR = Path("MNIST_Experiment_Results")
OUTPUT_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 512
EPOCHS = 20  # Required by the teacher
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Dataset --------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# With DATA_ROOT=D:\data, torchvision uses D:\data\MNIST\raw and processed.
if not (DATA_ROOT / "MNIST").exists():
    raise FileNotFoundError(
        f"MNIST folder was not found at {DATA_ROOT / 'MNIST'}. "
        "Check that your dataset is extracted under D:\\data\\MNIST."
    )

train_dataset = datasets.MNIST(
    root=str(DATA_ROOT), train=True, download=False, transform=transform
)
test_dataset = datasets.MNIST(
    root=str(DATA_ROOT), train=False, download=False, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# -------------------- CNN model: teacher's architecture --------------------
class Example4_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(2000, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = Example4_1().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# -------------------- Training and evaluation --------------------
start_time = time.time()
history = []

print("=" * 60)
print("MNIST HANDWRITTEN DIGIT RECOGNITION")
print(f"Device: {DEVICE} | Training images: {len(train_dataset)} | Test images: {len(test_dataset)}")
print(f"Epochs: {EPOCHS} | Batch size: {BATCH_SIZE}")
print("=" * 60)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)

    average_loss = total_loss / total_train
    train_accuracy = 100.0 * correct_train / total_train

    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct_test += (predictions == labels).sum().item()
            total_test += labels.size(0)

    test_accuracy = 100.0 * correct_test / total_test
    history.append({
        "epoch": epoch + 1,
        "training_loss": average_loss,
        "training_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    })
    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )

# -------------------- Save model and results --------------------
model_path = OUTPUT_DIR / "mnist_cnn_model.pth"
torch.save(model.state_dict(), model_path)

with open(OUTPUT_DIR / "training_results.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

# Training loss and accuracy graph
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
epoch_numbers = [item["epoch"] for item in history]
axes[0].plot(epoch_numbers, [item["training_loss"] for item in history], marker="o")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)
axes[1].plot(epoch_numbers, [item["test_accuracy"] for item in history], marker="o")
axes[1].set_title("Test Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "training_graphs.png", dpi=200)
plt.close(fig)

# Save a visual sample of predictions from the test set
model.eval()
images, labels = next(iter(test_loader))
with torch.no_grad():
    predictions = model(images.to(DEVICE)).argmax(dim=1).cpu()

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for index, axis in enumerate(axes.flat):
    image = images[index].squeeze().numpy() * 0.3081 + 0.1307  # Undo normalization
    axis.imshow(image, cmap="gray")
    axis.set_title(f"Actual: {labels[index].item()} | Pred: {predictions[index].item()}")
    axis.axis("off")
fig.suptitle("MNIST Test Predictions")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "prediction_samples.png", dpi=200)
plt.close(fig)

elapsed = time.time() - start_time
final = history[-1]
summary = (
    "MNIST HANDWRITTEN DIGIT RECOGNITION - EXPERIMENT RESULTS\n"
    f"Device: {DEVICE}\n"
    f"Training samples: {len(train_dataset)}\n"
    f"Testing samples: {len(test_dataset)}\n"
    f"Epochs: {EPOCHS}\n"
    f"Batch size: {BATCH_SIZE}\n"
    f"Final training loss: {final['training_loss']:.4f}\n"
    f"Final training accuracy: {final['training_accuracy']:.2f}%\n"
    f"Final test accuracy: {final['test_accuracy']:.2f}%\n"
    f"Total running time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)\n"
)
(OUTPUT_DIR / "experiment_summary.txt").write_text(summary, encoding="utf-8")

print("\n" + summary)
print(f"All outputs have been saved in: {OUTPUT_DIR.resolve()}")
