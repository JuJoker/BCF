import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.in_channels + 4
        self.kernel_size = 25
        self.padding = (self.kernel_size - 1) // 2
        self.hidden_size = 64
        self.cnn = nn.Conv1d(in_channels=self.in_channels, out_channels=1, stride=1, kernel_size=self.kernel_size, padding=self.padding)
        self.lstm = nn.LSTM(input_size=self.seq_len, hidden_size=self.hidden_size, num_layers=3, batch_first=True)
        self.fcn = nn.Linear(in_features=self.hidden_size, out_features=self.pred_len)


    def forward(self, true_target, true_univariate, true_time_mask):
        true_target = torch.unsqueeze(true_target, dim=2)
        # concatenate the target data and features data
        data = torch.cat((true_target, true_univariate, true_time_mask), dim=2)  # [batch, seq_len, feature_num+target_num]
        cnn_in = data.permute(0,2,1)
        cnn_out = self.cnn(cnn_in)
        lstm_out, _ = self.lstm(cnn_out)
        res = self.fcn(lstm_out)
        res = res.permute(0,2,1).squeeze()
        return res