import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class GRU(nn.Module):
    def __init__(self, hidden_dim, encode_dim, label_size):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.encode_dim = encode_dim
        self.label_size = label_size

        self.batch_size = 64
        # gru 双向gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear nn.Softmax(dim=None) 归一化
        self.linear = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.label_size), nn.Dropout(p=0.2) , nn.Softmax(dim=None))

        # hidden
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def forward(self, x):
        self.batch_size = len(x)
        self.hidden = self.init_hidden()
        x = torch.tensor(x , dtype=torch.float32)
        x = x.view(-1, 1, self.encode_dim)

        # gru 重新设置了隐藏层(输出)的大小 ,初始化为100
        gru_out, hidden = self.bigru(x, self.hidden)
        gru_out = torch.transpose(gru_out, 1, 2)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)

        # linear
        y = self.linear(gru_out)
        return y

if __name__ == '__main__':
    model = GRU(100 , 5 ,2)
    print(model)
    data = [[1,0,2,1,1] , [2,2,2,2,2] , [3,2,2,2,3]]

    print(model(data))