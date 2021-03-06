from torch import nn
import torch

import torch.nn.functional as F


class lstm(nn.Module):
    def __init__(self, input_size=128, hidden_size=100, output_size=2, num_layer=2):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        # self.layer2 = nn.Linear(hidden_size, output_size)

        # linear nn.Softmax(dim=None) 归一化
        self.linear = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Dropout(p=0.2))

    def forward(self, x):
        # (batch_size , seq_length , hidden_size)
        x, _ = self.layer1(x)
        # 转换维度
        # (batch_size , hidden_size, seq_length)
        x = torch.transpose(x , 1 , 2)
        # 降维，去除seq_length
        # (batch_size , hidden_size)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    model = lstm(2, 4, 2, 2)

    x = torch.randn(2, 3, 2)
    x = [[[1, 1],
          [2, 2],
          [3, 3]],
         [[2, 1],
          [3, 4],
          [1, 3]]]
    x = torch.FloatTensor(x)
    print(x.shape)
    print(model)
    output = model(x)
    print(output)
    _, predicted = torch.max(output, 1)
    print(predicted)
