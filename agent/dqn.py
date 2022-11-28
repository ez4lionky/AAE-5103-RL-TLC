# from policy.dqn.encoder import TLC_Encoder
import copy
import numpy as np
from agent.MLP import MLP, MLP_rnn
import torch.nn as nn
import torch
from agent.cnn import CNNBase
# from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F


class DQNNet(nn.Module):
    def __init__(self, hidden_size, obs_length, map_shape, action_space, device):
        super(DQNNet, self).__init__()
        self.device = device
        self.fc0 = MLP(obs_length, hidden_size)
        self.hidden = MLP(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_space)
        self.to(device)

    def forward(self, obs, obs_map, edges, edges_feature):
        """
        obs_map (_type_): [batch, n_nodes, 12+4]
        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        obs_map = torch.tensor(obs_map, device=self.device, dtype=torch.float)
        # edges = torch.tensor(edges, device=self.device, dtype=torch.long)
        # edges_feature = torch.tensor(edges_feature, device=self.device, dtype=torch.float)
        # batchsize, n_nodes = obs.shape[0], obs.shape[1]

        obs_out = self.fc0(obs)  # batch, dims
        # obs = obs.reshape(batchsize, n_nodes, -1)  # batch, n_nodes, dims
        feats = obs_out  # feats = torch.concatenate([obs, obs_map])
        hidden_state = self.hidden(feats)
        out = self.out(hidden_state)
        return out, [None]


class DQNMapNet(nn.Module):
    def __init__(self, hidden_size, obs_length, map_shape, action_space, device):
        super(DQNMapNet, self).__init__()
        self.device = device
        self.map_input = CNNBase(hidden_size, map_shape)

        self.hidden = MLP(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_space)  # action_space = 8
        self.to(device)

    def forward(self, obs, obs_map, edges, edges_feature):
        """
        obs_map (_type_): [batch, n_nodes, 12+4]
        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        obs_map = torch.tensor(obs_map, device=self.device, dtype=torch.float)
        # edges = torch.tensor(edges, device=self.device, dtype=torch.long)
        # edges_feature = torch.tensor(edges_feature, device=self.device, dtype=torch.float)
        # batchsize, n_nodes = obs.shape[0], obs.shape[1]

        obs_map_out = self.map_input(obs_map)  # batch, dims
        # obs = obs.reshape(batchsize, n_nodes, -1)  # batch, n_nodes, dims
        feats = obs_map_out  # feats = torch.concatenate([obs, obs_map])
        hidden_state = self.hidden(feats)
        out = self.out(hidden_state)
        return out, [None]
