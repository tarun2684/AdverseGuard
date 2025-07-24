import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import logging


class SimpleMNISTCNN(nn.Module):
    def __init__(self):
        super(SimpleMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)

        # Dummy input to calculate the flatten size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            x = self.pool(F.relu(self.conv2(F.relu(self.conv1(dummy_input)))))
            self.flatten_dim = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # dynamic flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
def load_model(pretrained=False):
    model = SimpleMNISTCNN()
    return model
