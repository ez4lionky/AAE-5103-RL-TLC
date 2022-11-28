import torch
import torch.nn.functional as F
from torch import nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MLP_rnn(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP_rnn, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        # self.layer_norm = nn.LayerNorm(out_features)
        # self.batch_norm = nn.BatchNorm1d(out_features)

        # init_method = nn.init.orthogonal_
        # gain = nn.init.calculate_gain('relu')

        # def init_(m):
        #     return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        # self.linear = init_(self.linear)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        # hidden_states = self.layer_norm(hidden_states)
        # hidden_states = self.batch_norm(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        # self.layer_norm = nn.LayerNorm(out_features)
        # self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        # hidden_states = self.layer_norm(hidden_states)
        # hidden_states = self.batch_norm(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states
