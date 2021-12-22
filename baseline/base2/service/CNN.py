import torch.nn as nn
import torch

# CNN 主要还是用来做预测的
# 三星
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self , vocab_size , embedding_dim , pretrained_weight):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # input_channels:输入通道，文本应该为1，图片可能有3通道，即为RGB
        # out_channels:输出通道，即为filter_num
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(16, 128), padding=0) ,
            nn.LeakyReLU(),
            nn.MaxPool2d((4 , 1) , (2,1))
        )
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(1728, 16)
        self.fc = nn.Linear(16 , 2)

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, 128))
        return zeros


    def forward(self, x):
        lens = [len(i) for i in x]
        max_len = max(lens)
        seq = []
        for i in range(len(x)):
            seq.append(self.embedding(torch.LongTensor(x[i])))
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
        out = torch.cat(seq)
        out = out.view(len(x) , max_len , -1)
        print(out.shape)

        out = out.unsqueeze(1)

        print(out.shape)

        out = self.conv1(out)
        print(out.shape)
        out = out.view(len(x), -1)
        print(out.shape)
        out = self.dropout(out)
        out = self.sigmoid(out)
        out = self.linear1(out)
        return self.fc(out)


if __name__ == '__main__':
    model = CNN()
    t1 = torch.rand(2, 1 , 50 , 128)
    print(model(t1))