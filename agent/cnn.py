import math
import torch.nn as nn


# CNN Modules and utils.
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_cnn_out_size(input_height, input_width, kernel_size, stride, dilation=1, padding=0):
    out_height = math.floor((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    out_width = math.floor((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return out_height, out_width


class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU, kernel_size=[5, 3], stride=1):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]  # 2
        input_height = obs_shape[1]  # 12
        input_width = obs_shape[2]  # 60

        self.conv2d1 = nn.Conv2d(in_channels=input_channel,
                                 out_channels=hidden_size // 2,
                                 kernel_size=kernel_size[0],
                                 stride=stride)
        self.conv2d2 = nn.Conv2d(in_channels=hidden_size // 2,
                                 out_channels=hidden_size,
                                 kernel_size=kernel_size[1],
                                 stride=stride)
        self.active_func = active_func
        self.flat = Flatten()
        out_height, out_width = get_cnn_out_size(input_height, input_width, kernel_size[0], stride)
        out_height, out_width = get_cnn_out_size(out_height, out_width, kernel_size[1], stride)
        in_channel = hidden_size * out_height * out_width
        self.fc1 = init_(nn.Linear(in_channel, 512))
        self.fc2 = init_(nn.Linear(512, hidden_size))

    def forward(self, x):
        x = self.active_func(self.conv2d1(x))
        x = self.active_func(self.conv2d2(x))
        # x = self.gap(x)
        x = self.flat(x)
        x = self.active_func(self.fc1(x))
        x = self.active_func(self.fc2(x))

        # x = self.cnn(x)
        return x


class CNNBase(nn.Module):
    def __init__(self, hidden_size, obs_shape):
        super(CNNBase, self).__init__()

        self._use_orthogonal = True
        self._use_ReLU = True
        self.hidden_size = hidden_size

        self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        x = self.cnn(x)
        return x
