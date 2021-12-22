from torch import nn
import torch

import torch.nn.functional as F
from torch.autograd import Variable


class lstm(nn.Module):
    def __init__(self, input_size=128, hidden_size=100, output_size=2, num_layer=2):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.input_size))
        return zeros

    def forward(self, x):
        # (batch_size , seq_length , hidden_size)
        x, _ = self.layer1(x)
        # 转换维度
        # (batch_size , hidden_size, seq_length)
        x = torch.transpose(x , 1 , 2)
        # 降维，去除seq_length
        # (batch_size , hidden_size)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.layer2(x)
        return x


model = lstm(2, 4, 2, 2)

x = torch.randn(2 , 3 , 2)
x = [[[1 ,1],
      [2 ,2],
      [3 ,3]],
     [[2,1],
      [3,4],
      [1,3]]]
x = torch.FloatTensor(x)
print(x.shape)
print(model)
output = model(x)
print(output)
_, predicted = torch.max(output, 1)
print(predicted)