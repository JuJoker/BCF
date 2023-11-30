import torch
import torch.nn as nn


class Model(nn.Module):

     def __init__(self, configs):

         super(Model, self).__init__()
         self.seq_len = configs.seq_len
         self.pred_len = configs.pred_len
         self.hidden_size = 64
         self.encoder = nn.GRU(input_size=self.seq_len, hidden_size=self.hidden_size, num_layers=3,batch_first=True)
         self.decoder = nn.GRU(input_size=1, hidden_size=self.hidden_size, num_layers=3, batch_first=True)
         self.flatten = nn.Flatten()
         self.fcn = nn.Linear(self.hidden_size, self.pred_len)

     def forward(self, true_target, true_univariate, true_time_mask):
         true_target = torch.unsqueeze(true_target, dim=2)
         # concatenate the target data and features data
         data = torch.cat((true_target, true_univariate, true_time_mask),
                          dim=2)  # [batch, seq_len, feature_num+target_num]
         enc_in = data.permute(0, 2, 1)
         enc_out, hidden_state = self.encoder(enc_in)
         dec_in = enc_out[:,-1:,-1:]
         dec_out, _ = self.decoder(dec_in, hidden_state)
         fcn_in = self.flatten(dec_out)
         res = self.fcn(fcn_in)
         return res
