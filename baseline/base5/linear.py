import torch
import torch.nn as nn
import numpy as np


class Linear(nn.Module):
    def __init__(self, hidden_dim, label_size):
        super(Linear, self).__init__()
        self.label_size = label_size
        self.hidden_dim = hidden_dim
        # linear nn.Softmax(dim=None) 归一化
        self.linear = nn.Sequential(nn.Linear(self.hidden_dim, self.label_size), nn.Dropout(p=0.2) , nn.Softmax(dim=None))

    def forward(self, x):
        # linear
        f_len = len(x[0])
        x = np.array(x)
        x.astype(float)
        x = torch.from_numpy(x)
        x = x.view(-1, f_len)
        x = x.to(torch.float32)
        y = self.linear(x)
        return y

if __name__ == '__main__':
    model = Linear(5 ,2)
    print(model)
    data = [[1,0,2,1,1] , [2,2,2,2,2] , [3,2,2,2,3]]

    print(model(data))