import torch.nn as nn
import torch.nn.functional as F

# Define an Optimized Lightweight CNN Model with <25,000 parameters
class UltraSmallCNN(nn.Module):
    def __init__(self):
        super(UltraSmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(7)  # BatchNorm for first convolutional layer
        self.conv2 = nn.Conv2d(7, 14, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(14)  # BatchNorm for second convolutional layer
        self.fc1 = nn.Linear(14 * 7 * 7, 32)
        self.bn3 = nn.BatchNorm1d(32)  # BatchNorm for fully connected layer
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.leaky_relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))  # Conv1 + BatchNorm + ReLU + MaxPool
        x = F.leaky_relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))  # Conv2 + BatchNorm + ReLU + MaxPool
        x = x.view(-1, 14 * 7 * 7)  # Flatten
        x = F.leaky_relu(self.bn3(self.fc1(x)))  # FC1 + BatchNorm + ReLU
        x = self.fc2(x)
        return x
