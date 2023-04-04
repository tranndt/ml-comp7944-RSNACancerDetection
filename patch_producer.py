import torch
import torch.nn as nn

class PatchProducer(nn.Module):
    def __init__(self, input_dim=36, patch_len=16, dropout=0.2, channels=3):
        super(PatchProducer, self).__init__()
        self.channels=channels
        self.patch_len = patch_len
        self.fc1 = nn.Linear(input_dim, 128) # input layer
        self.fc2 = nn.Linear(128, 256) # hidden layer 1
        self.fc3 = nn.Linear(256, 512) # hidden layer 2
        self.fc4 = nn.Linear(512, patch_len*patch_len*channels) # output layer
        self.dropout = nn.Dropout(dropout) # dropout layer

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x.reshape(x.shape[0], self.channels, self.patch_len, self.patch_len)
