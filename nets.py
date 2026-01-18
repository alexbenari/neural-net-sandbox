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
        self.fc2a = nn.Linear(900, 900)
        self.fc2b = nn.Linear(900, 900)
        self.fc2c = nn.Linear(900, 900)
        self.fc2d = nn.Linear(900, 900)
        self.fc2e = nn.Linear(900, 900)
        self.fc2f = nn.Linear(900, 900)
        self.fc2g = nn.Linear(900, 900)
        self.fc2h = nn.Linear(900, 900)
        self.fc2i = nn.Linear(900, 900)
        self.fc2j = nn.Linear(900, 900)
        self.fc3 = nn.Linear(900, 300)
        self.fc4 = nn.Linear(300, 200)
        self.fc5 = nn.Linear(200, 20)
        self.fc6 = nn.Linear(20, 1)
        #self.fc7 = nn.Linear(900, 1)
        nn.init.constant_(self.fc6.bias, 96.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2a(x))
        x = torch.relu(self.fc2b(x))
        x = torch.relu(self.fc2c(x))
        '''x = torch.relu(self.fc2d(x))
        x = torch.relu(self.fc2e(x))
        x = torch.relu(self.fc2f(x))
        x = torch.relu(self.fc2g(x))
        x = torch.relu(self.fc2h(x))
        x = torch.relu(self.fc2i(x))
        x = torch.relu(self.fc2j(x))'''
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
