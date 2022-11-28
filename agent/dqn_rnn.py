# from policy.dqn.encoder import TLC_Encoder
import copy
import numpy as np
from agent.MLP import MLP_rnn
import torch.nn as nn
import torch
from agent.cnn import CNNBase
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class TLC_DQN(nn.Module):
    def __init__(self, hidden_size, obs_length, map_shape, action_space, device, dueling=True):
        super(TLC_DQN, self).__init__()
        self.device = device
        self.dueling = dueling

        self.nbrs_input = MLP_rnn(obs_length + 4, hidden_size)  # obs_space = 12
        self.map_input = CNNBase(hidden_size, map_shape)

        gat_head_num = 5
        edge_feature_dim = 4
        self.gat_head_num = gat_head_num
        self.gat = GATv2Conv(in_channels=hidden_size, out_channels=hidden_size, heads=gat_head_num,
                             concat=False)  # edge_dim=edge_feature_dim
        self.GRU = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # init_method = nn.init.orthogonal_
        # gain = nn.init.calculate_gain('relu')

        # def init_(m):
        #     return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.hidden = MLP_rnn(hidden_size, hidden_size)
        if not self.dueling:
            self.out = nn.Linear(hidden_size, action_space)  # action_space = 8
            # self.out = init_(self.out)
        else:
            self.out = nn.Linear(hidden_size, 1)
            self.adv = nn.Linear(hidden_size, action_space)
            # self.out = init_(self.out)
            # self.adv = init_(self.adv)
        self.to(device)

    def forward(self, obs, obs_map, edges, edges_feature):
        """
        obs_map (_type_): [batch, n_nodes, seq_len, 12+4]
        """
        self.GRU.flatten_parameters()
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        obs_map = torch.tensor(obs_map, device=self.device, dtype=torch.float)
        edges = torch.tensor(edges, device=self.device, dtype=torch.long)
        edges_feature = torch.tensor(edges_feature, device=self.device, dtype=torch.float)
        batchsize, n_nodes = obs.shape[0], obs.shape[1]

        target_emb = self.map_input(obs_map).unsqueeze(1)

        nbrs_input = obs[:, 1:, :, :]  # batch, n_nodes-1, seq_len, dims
        nbrs_input = nbrs_input.reshape(batchsize * (n_nodes - 1),
                                        *nbrs_input.shape[2:])  # batch*(n_nodes-1), seq_len, dims
        nbrs_emb = self.nbrs_input(nbrs_input)  # B, T, C
        _, nbrs_rnn = self.GRU(nbrs_emb)
        nbrs_rnn = nbrs_rnn.squeeze(0).reshape(batchsize, n_nodes - 1, -1)  # batch*(n_nodes-1), dims
        obs = torch.cat((target_emb, nbrs_rnn), dim=1)  # batch, n_nodes, dims

        obs_gat = []
        att_weights_batch = []  # (edge_num, head_num)
        for i in range(batchsize):
            gat_f, (edge_idx, att_weigts) = self.gat(x=obs[i], edge_index=edges[i], return_attention_weights=True)
            # edge_attr=edges_feature[i]
            obs_gat.append(gat_f[0])  # 1, head*dims

            att_weigts = np.mean(att_weigts[:-n_nodes + 1, :].detach().cpu().numpy(), axis=1)
            att_weights_batch.append(att_weigts)

        obs_gat = torch.stack(obs_gat)  # batch, head*dims

        hidden_state = self.hidden(obs_gat)
        out = self.out(hidden_state)
        if self.dueling:
            adv = self.adv(hidden_state)
            out = out + (adv - adv.mean(axis=-1, keepdim=True))
        return out, att_weights_batch  # att_weights_batch
