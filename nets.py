import torch
import torch.nn as nn


class SimpleMLPForInt(nn.Module):
    def __init__(self):
        super(SimpleMLPForInt, self).__init__()
        self.fc1 = nn.Linear(1, 30)
        self.fc2 = nn.Linear(30   , 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class SimpleMLPForDigits(nn.Module):
    def __init__(self):
        super(SimpleMLPForDigits, self).__init__()
        self.fc1 = nn.Linear(9, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class SimpleMLPForBinary(nn.Module):
    def __init__(self):
        super(SimpleMLPForBinary, self).__init__()
        self.fc1 = nn.Linear(30, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
