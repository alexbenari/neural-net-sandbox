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
        self.fc2 = nn.Linear(30, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 200)
        self.fc5 = nn.Linear(200, 20)
        self.fc6 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class SimpleMLPForDigit1H(nn.Module):
    def __init__(self):
        super(SimpleMLPForDigit1H, self).__init__()
        self.fc1 = nn.Linear(90, 900)
        self.fc2 = nn.Linear(900, 900)
        self.fc3 = nn.Linear(900, 600)
        self.fc4 = nn.Linear(600, 200)
        self.fc5 = nn.Linear(200, 20)
        self.fc6 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
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
