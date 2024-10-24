import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple model that takes up around 1GB of memory
# To estimate the size: A tensor with float32 takes 4 bytes per element
# 1 GB = 1e9 bytes. So we need approximately (1e9 / 4) elements = 2.5e8 parameters

class SimpleLargeModel(nn.Module):
    def __init__(self, input_size=20000, hidden_size=10000, output_size=5000):
        super(SimpleLargeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = SimpleLargeModel().to(device)

# Print out the number of parameters (this gives an idea of model size)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model has {num_params} parameters")

# Optimizer and loss function (for simulation purposes)
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Simulate training with dummy data
for epoch in range(30):  # small number of epochs just for simulation
    print(f"Epoch {epoch + 1}")
    model.train()

    # Create dummy data simulating a batch size of 64
    inputs = torch.randn(64, 20000).to(device)  # Input size should match the model's input
    targets = torch.randn(64, 5000).to(device)  # Target size matches model's output

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item()}")

# Checking if the model is on GPU
if next(model.parameters()).is_cuda:
    print("Model is using GPU!")
else:
    print("Model is using CPU.")
