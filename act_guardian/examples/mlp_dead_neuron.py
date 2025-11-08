import torch
import torch.nn as nn
from act_guardian import act_guardian

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MLP()
x = torch.randn(32, 10) * 0.001  # simulate low activation
y = torch.randn(32, 1)
guard_fn = act_guardian(model)

for i in range(50):
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    guard_fn()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    model.zero_grad()
    if i % 10 == 0:
        print(f"Step {i} | Loss: {loss.item():.4f} | Act norm: {model.fc1(x).norm():.6f}")
