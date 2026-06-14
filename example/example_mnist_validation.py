import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Config ---
BATCH_SIZE   = 1
EPOCHS       = 1
LR           = 1e-3
LOG_INTERVAL   = 1   # print training stats every N steps
TEST_INTERVAL  = 10000  # run test-set evaluation every N steps
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
])

train_dataset = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# --- Model ---
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14x14 -> 14x14
        self.pool  = nn.MaxPool2d(2)                              # halves spatial dims
        self.drop  = nn.Dropout(0.25)

        # Classifier head
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (B, 64,  7,  7)
        x = self.drop(x)
        x = x.flatten(1)                      # -> (B, 3136)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                       # logits, no softmax (CrossEntropyLoss handles it)
        return x

model = ConvNet().to(DEVICE)

# --- Training ---
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = nn.CrossEntropyLoss()

# --- Evaluation ---
def evaluate(step):
    model.eval()
    total_loss, correct = 0.0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * len(images)
            correct    += (logits.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(test_dataset)
    accuracy = 100.0 * correct / len(test_dataset)
    print(f"Eval {step:5d} | Test  loss: {avg_loss:.4f} | Test  acc: {accuracy:.2f}%")

# --- Run ---
global_step = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss, epoch_correct = 0.0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)
        loss   = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1
        epoch_loss += loss.item() * len(images)
        epoch_correct += (logits.argmax(1) == labels).sum().item()

        if global_step % LOG_INTERVAL == 0:
            steps_in_epoch = global_step - (epoch - 1) * len(train_dataset)
            avg_loss = epoch_loss / steps_in_epoch
            accuracy = 100.0 * epoch_correct / steps_in_epoch
            print(f"Step {global_step:5d}/{len(train_dataset):5d} | Train loss: {avg_loss:.4f} | Train acc: {accuracy:.2f}%")

        if global_step % TEST_INTERVAL == 0:
            evaluate(global_step)

    # Epoch-end summary
    avg_loss = epoch_loss / len(train_dataset)
    accuracy = 100.0 * epoch_correct / len(train_dataset)
    print(f"Epoch {epoch:2d} (step {global_step:5d}) | Train loss: {avg_loss:.4f} | Train acc: {accuracy:.2f}%")
    evaluate(global_step)

torch.save(model.state_dict(), "checkpoint.pt")
print(f"wrote checkpoint to disk")