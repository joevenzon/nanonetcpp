import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- Data ---
X = torch.linspace(-torch.pi, torch.pi, 1000).unsqueeze(1)
y = torch.sin(X)

# --- Model ---
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)

# --- Training ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(5000):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

# --- Write outputs ---
torch.save(model.state_dict(), "checkpoint.pt")
print(f"wrote checkpoint to disk")