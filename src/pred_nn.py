import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class PredRNN(nn.Module):
    def __init__(self, input_size=1, initialization_size=21, hidden_size=128, output_size=2, device='cuda'):
        super(PredRNN, self).__init__()
        self.device = device
        self.count=2
        self.hidden_size = hidden_size
        self.init_net = nn.Sequential(
            nn.Linear(initialization_size, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size*self.count),
            nn.Tanh()
        )
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Linear(hidden_size*self.count, output_size)

    def forward(self, input, imgs, count, hidden=None):
        hidden_init = self.init_net(input)
        hidden_init = hidden_init.reshape((self.count, hidden_init.shape[0], self.hidden_size))
        cell_state = torch.zeros_like(hidden_init)
        hidden = (hidden_init, cell_state)
        packed_input = pack_padded_sequence(imgs, count, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell_state) = self.lstm(packed_input, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.output_net(output)
        output = output[:, -1, :]
        return output
