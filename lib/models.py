import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class SimpleFFDQN(nn.Module):
    def __init__(self, obs_len, actions_n):
        super(SimpleFFDQN, self).__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

class SimpleLSTM(nn.Module):
    def __init__(self, status_size, input_size=5, n_hidden=512, n_layers=2, rnn_drop_prob=0.5, fc_drop_prob=0.3, actions_n=3,
                 train_on_gpu=True, batch_first=True):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.rnn_drop_prob = rnn_drop_prob
        self.fc_drop_prob = fc_drop_prob
        self.actions_n = actions_n
        self.train_on_gpu = train_on_gpu
        if self.train_on_gpu:
            self.device = torch.device("cuda")
        self.batch_first = batch_first
        self.batch_size = None
        self.status_size = status_size

        self.lstm = nn.LSTM(self.input_size, self.n_hidden, self.n_layers, dropout=self.rnn_drop_prob, batch_first=self.batch_first)

        # for state value
        self.fc_val = nn.Sequential(
            nn.Linear(self.n_hidden + self.status_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # for action value
        self.fc_adv = nn.Sequential(
            nn.Linear(self.n_hidden + self.status_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.actions_n)
        )

        self.hidden = None

    def preprocessor(self, x): # dtype = torch.float32 must be set
        data = None
        status = None
        if len(x) == 1:
            data = torch.tensor(np.expand_dims(x[0].data, 0), dtype=torch.float32).to(self.device)   # data.shape = (1, 20, 5)
            status = torch.tensor(x[0].status, dtype=torch.float32).to(self.device)                  # status.shape = (1,2)
        elif len(x) > 1:
            data_shape = np.insert(x[0].data.shape, 0, self.batch_size)       # data.shape = (batch_size, 20, 5)
            data_arr = np.ndarray(shape=data_shape, dtype=np.float32)
            status_shape = np.array([self.batch_size, x[0].status.shape[1]])  # status.shape = (batch_size,2)
            status_arr = np.ndarray(shape=status_shape, dtype=np.float32)
            for idx, exp in enumerate(x):
                data_arr[idx, :, :] = np.expand_dims(x[idx].data, 0)
                status_arr[idx, :] = x[idx].status
                data = torch.tensor(data_arr, dtype=torch.float32).to(self.device)
                status = torch.tensor(status_arr, dtype=torch.float32).to(self.device)
        return data, status

    def forward(self, x):
        data, status = self.preprocessor(x)
        self.hidden = tuple([each.data for each in self.hidden])
        self.lstm.flatten_parameters()
        r_output, self.hidden = self.lstm(data, self.hidden)
        # only need the last cell output
        output = r_output[:,-1,:]
        output = output.view(self.batch_size, -1)
        output = torch.cat([output, status], dim=1) # shape = o(batch_size, 512) + status(batch_size, 2) = (batch_size,514)

        val = self.fc_val(output)
        adv = self.fc_adv(output)

        return val + adv - adv.mean(dim=1, keepdim=True)

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        self.batch_size = batch_size
        weight = next(self.parameters()).data

        if (self.train_on_gpu):
            self.hidden = (weight.new_zeros([self.n_layers, batch_size, self.n_hidden]).cuda(),
                      weight.new_zeros([self.n_layers, batch_size, self.n_hidden]).cuda())
        else:
            self.hidden = (weight.new_zeros([self.n_layers, batch_size, self.n_hidden]),
                      weight.new_zeros([self.n_layers, batch_size, self.n_hidden]))

class DoubleLSTM(nn.Module):
    def __init__(self, env, n_hidden=256, n_layers=2, rnn_drop_prob=0.2, fc_drop_prob=0.2, actions_n=3,
                 train_on_gpu=True, batch_first=True):
        super(DoubleLSTM, self).__init__()
        self.env = env
        self.price_input_size = self.env.price_size[1]
        self.trend_input_size = self.env.trend_size[1]
        self.status_size = self.env.status_size[1]
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.rnn_drop_prob = rnn_drop_prob
        self.fc_drop_prob = fc_drop_prob
        self.actions_n = actions_n
        self.train_on_gpu = train_on_gpu
        if self.train_on_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.batch_first = batch_first
        self.batch_size = None
        self.img1_output_size = 0

        self.price_lstm = nn.LSTM(self.price_input_size, self.n_hidden, self.n_layers, dropout=self.rnn_drop_prob,
                            batch_first=self.batch_first).to(self.device)
        self.trend_lstm = nn.LSTM(self.trend_input_size, self.n_hidden, self.n_layers, dropout=self.rnn_drop_prob,
                            batch_first=self.batch_first).to(self.device)

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=64, kernel_size=10),
            nn.BatchNorm1d(num_features=64),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=10),
            nn.BatchNorm1d(num_features=32),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10),
            nn.BatchNorm1d(num_features=32),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=10),
            nn.BatchNorm1d(num_features=16),
            nn.MaxPool1d(kernel_size=2)
        ).to(self.device)
        self.get_img1_output_size()

        # for state value
        self.fc_val = nn.Sequential(
            nn.Linear(self.n_hidden*2 + self.img1_output_size + self.status_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        # for action value
        self.fc_adv = nn.Sequential(
            nn.Linear(self.n_hidden*2 + self.img1_output_size + self.status_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=self.fc_drop_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.actions_n)
        ).to(self.device)

        self.hidden_p = None
        self.hidden_t = None

    def get_img1_output_size(self):
        dump_input = torch.randn((1, self.env.image_sizes[0][0], self.env.image_sizes[0][1])).permute(0, 2, 1).to(self.device)
        out_ = self.conv1d(dump_input)
        out = out_.permute(0, 2, 1).contiguous().view(1, -1)
        self.img1_output_size = out.shape[1]

    def preprocessor(self, states):  # dtype = torch.float32 must be set

        price_data = None
        trend_data = None
        image1_data = None
        status_data = None

        if len(states) == 1:
            price_data = torch.tensor(np.expand_dims(states[0].price, 0), dtype=torch.float32, device=self.device)  # data.shape = (1, 40, 4)
            trend_data = torch.tensor(np.expand_dims(states[0].trend, 0), dtype=torch.float32, device=self.device)  # data.shape = (1, 40, 4)
            image1_data = torch.tensor(np.expand_dims(states[0].image1, 0), dtype=torch.float32, device=self.device).permute(0, 2, 1) # data.shape = (1, 100, 80)
            status_data = torch.tensor(states[0].status, dtype=torch.float32, device=self.device)  # status.shape = (1,2)
        elif len(states) > 1:
            price_data_shape = (len(states), states[0].price.shape[0], states[0].price.shape[1])  # data.shape = (batch_size, 40, 4)
            trend_data_shape = (len(states), states[0].trend.shape[0], states[0].trend.shape[1])  # data.shape = (batch_size, 40, 4)
            image1_data_shape = (len(states), states[0].image1.shape[0], states[0].image1.shape[1])  # image1.shape = (batch_size, 80, 100)
            status_data_shape = (len(states), states[0].status.shape[1])  # status.shape = (batch_size,2)
            price_data_arr = np.zeros(shape=price_data_shape, dtype=np.float32)
            trend_data_arr = np.zeros(shape=trend_data_shape, dtype=np.float32)
            image1_data_arr = np.zeros(shape=image1_data_shape, dtype=np.float32)
            status_data_arr = np.zeros(shape=status_data_shape, dtype=np.float32)
            for idx, exp in enumerate(states):
                price_data_arr[idx, :, :] = np.expand_dims(states[idx].price, 0)
                trend_data_arr[idx, :, :] = np.expand_dims(states[idx].trend, 0)
                image1_data_arr[idx, :, :] = np.expand_dims(states[idx].image1, 0)
                status_data_arr[idx, :] = states[idx].status
            price_data = torch.tensor(price_data_arr, dtype=torch.float32, device=self.device)
            trend_data = torch.tensor(trend_data_arr, dtype=torch.float32, device=self.device)
            image1_data = torch.tensor(image1_data_arr, dtype=torch.float32, device=self.device).permute(0, 2, 1)  # image1.shape = (batch_size, 100, 80)
            status_data = torch.tensor(status_data_arr, dtype=torch.float32, device=self.device)
        return price_data, trend_data, image1_data, status_data

    def forward(self, x):
        price, trend, image1, status = self.preprocessor(x)

        self.hidden_p = tuple([each.data for each in self.hidden_p])
        self.price_lstm.flatten_parameters()
        price_output, self.hidden_p = self.price_lstm(price, self.hidden_p)

        self.hidden_t = tuple([each.data for each in self.hidden_t])
        self.trend_lstm.flatten_parameters()
        trend_output, self.hidden_t = self.trend_lstm(trend, self.hidden_t)

        img1_out_ = self.conv1d(image1)    # (batch_size, feature_size, output_size) # feature_size = filter_number
        img1_out = img1_out_.contiguous().view(self.batch_size, -1)

        # only need the last cell output
        price_output_ = price_output[:,-1,:]
        trend_output_ = trend_output[:, -1, :]
        price_output = price_output_.view(self.batch_size, -1)
        trend_output = trend_output_.view(self.batch_size, -1)
        # cat output: shape = po(batch_size, 256) + to(batch_size, 256) + status(batch_size, 2) = (batch_size,514)
        output_ = torch.cat((price_output, trend_output), dim=1)
        output_ = torch.cat((output_, img1_out), dim=1)
        output = torch.cat((output_, status), dim=1)

        val = self.fc_val(output)
        adv = self.fc_adv(output)

        return val + adv - adv.mean(dim=1, keepdim=True)

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        self.batch_size = batch_size
        if (self.train_on_gpu):
            weight = next(self.parameters()).data
            self.hidden_p = (weight.new_zeros([self.n_layers, batch_size, self.n_hidden], device=self.device),
                      weight.new_zeros([self.n_layers, batch_size, self.n_hidden], device=self.device))
            weight = next(self.parameters()).data
            self.hidden_t = (weight.new_zeros([self.n_layers, batch_size, self.n_hidden], device=self.device),
                           weight.new_zeros([self.n_layers, batch_size, self.n_hidden], device=self.device))
        else:
            weight = next(self.parameters()).data
            self.hidden_p = (weight.new_zeros([self.n_layers, batch_size, self.n_hidden]),
                      weight.new_zeros([self.n_layers, batch_size, self.n_hidden]))
            weight = next(self.parameters()).data
            self.hidden_t = (weight.new_zeros([self.n_layers, batch_size, self.n_hidden]),
                           weight.new_zeros([self.n_layers, batch_size, self.n_hidden]))

class DQNConv1D(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)


class DQNConv1DLarge(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1DLarge, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)
