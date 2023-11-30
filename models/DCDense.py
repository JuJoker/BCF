import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Dense(nn.Module):
    """Dense Layer"""

    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(Dense, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(self.in_dim, self.in_dim)
        self.drop = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, input):
        hidden = self.linear1(input)
        hidden = self.drop(hidden)
        res = self.linear2(hidden)
        return res


class Encoder(nn.Module):

    def __init__(self, in_channels=7, out_channels=7, kernel_size=1, stride=1, dropout=0.2):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=in_channels // 2,
                               kernel_size=kernel_size,
                               stride=stride)
        self.max_pool1 = nn.MaxPool1d(kernel_size=kernel_size,
                                     stride=stride)
        self.drop = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(in_channels=in_channels // 2,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)
        self.max_pool2 = nn.MaxPool1d(kernel_size=kernel_size,
                                      stride=stride)
        self.conv3 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)
        self.max_pool3 = nn.MaxPool1d(kernel_size=kernel_size,
                                      stride=stride)

    def forward(self, x):
        # x shape:[batch, feature_num + target_num, seq_len]
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.max_pool3(x)
        x = self.drop(x)
        return x


class Decoder(nn.Module):

    def __init__(self, configs):
        super(Decoder, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.batch_size = configs.batch_size
        self.dense1 = nn.Sequential(
            Dense(self.seq_len, self.pred_len, dropout=configs.dropout),
            Dense(self.pred_len, self.pred_len, dropout=configs.dropout),
            Dense(self.pred_len, self.pred_len, dropout=configs.dropout)
        )
        self.flatten = nn.Flatten()
        self.dense2 = nn.Sequential(
            Dense(self.seq_len, self.pred_len, dropout=configs.dropout),
            Dense(self.pred_len, self.pred_len, dropout=configs.dropout),
            Dense(self.pred_len, self.pred_len, dropout=configs.dropout)
        )
        self.out_dim = configs.out_channels + 2
        self.dense3 = nn.Sequential(
            Dense(self.pred_len*self.out_dim, self.pred_len, dropout=configs.dropout),
            Dense(self.pred_len, self.pred_len, dropout=configs.dropout),
            Dense(self.pred_len, self.pred_len, dropout=configs.dropout)
        )

    def forward(self, seasonal, trend, enc_out):
        data = torch.cat([seasonal, trend], dim=2)
        # map past seasonal components and trend components to the desired length for prediction.
        target = self.dense1(data.permute(0,2,1))
        # map the past features to the desired length for predict.
        features = self.dense2(enc_out)
        fea_tar = torch.cat((target, features), dim=1)      # [batch, 3, pred_len]
        fea_tar = self.flatten(fea_tar)
        res = self.dense3(fea_tar)      # [batch, pred_len]
        return res


class Model(nn.Module):
    """
    DCLinear Model
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # input time series length
        self.seq_len = configs.seq_len
        # predict time series length
        self.pred_len = configs.pred_len

        # Decomposition Kernel Size
        self.kernel_size = configs.kernel_size
        self.decompsition = series_decomp(self.kernel_size)

        # input channels
        self.in_channels = int(configs.in_channels + 4)
        # output channels
        self.out_channels = configs.out_channels

        self.encoder = Encoder(in_channels=self.in_channels, out_channels=self.out_channels, dropout=configs.dropout)
        self.decoder = Decoder(configs)

    def forward(self, true_target, true_univariate, true_time_mask):
        # dimensionality expansion
        true_target = torch.unsqueeze(true_target, dim=2)
        # decompose data
        seasonal_init, trend_init = self.decompsition(true_target)
        # concatenate the target data and features data
        enc_in = torch.cat((true_target, true_univariate, true_time_mask), dim=2)     # [batch, seq_len, feature_num+target_num]
        enc_in = enc_in.permute(0,2,1)      # [batch, feature_num+target_num, seq_len]
        enc_out = self.encoder(enc_in)
        dec_out = self.decoder(seasonal_init, trend_init, enc_out)
        return dec_out





