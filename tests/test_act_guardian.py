import torch
import torch.nn as nn
from act_guardian import act_guardian

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 3)
    def forward(self, x): return self.fc(x)

def test_guardian_revives():
    model = Net()
    x = torch.zeros(4, 5)  # zero activation
    guard_fn = act_guardian(model)
    
    model(x)  # trigger hook
    guard_fn()
    
    assert model._act_guard.norm() > 0  # revived!
