import torch.nn as nn

class PredNN(nn.Module):
    def __init__(self, input_dim=22):
        super(PredNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) # input layer
        self.fc2 = nn.Linear(128, 256) # hidden layer 1
        self.fc3 = nn.Linear(256, 128) # hidden layer 2
        self.fc4 = nn.Linear(128, 2) # output layer

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x
