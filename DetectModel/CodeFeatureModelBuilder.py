import torch
from torch.autograd import Variable

from CodeFeatureModel.model import CodeFeatureModel


class CodeFeatureModelBuilder:
    def __init__(self , train_data: list , train_label: list, embedding_dim: int,
                 hidden_dim: int, vocab_size:int, encode_dim:int, label_size:int,
                 batch_size: int, pretrained_weight=None):
        self.code_feature_model = CodeFeatureModel(embedding_dim, hidden_dim, vocab_size, encode_dim,
                                                   label_size, batch_size, pretrained_weight)
        self.train_data = train_data
        self.train_label = train_label
        self.EPOCHS = 15
        self.BATCH_SIZE = 64

    def train(self):
        parameters = self.code_feature_model.parameters()
        optimizer = torch.optim.Adamax(parameters)
        loss_function = torch.nn.CrossEntropyLoss()
        for epoch in range(self.EPOCHS):
            i = 0
            while i < len(self.train_data):
                cur_train_features = self.train_data[i:i + self.BATCH_SIZE]
                cur_train_labels = self.train_label[i:i + self.BATCH_SIZE]
                i += self.BATCH_SIZE
                optimizer.zero_grad()
                # 不足批次大小的要重新设置batchsize
                self.code_feature_model.batch_size = len(cur_train_labels)
                # 同时需要重置隐藏层
                self.code_feature_model.hidden = self.code_feature_model.init_hidden()
                output, predict = self.code_feature_model(cur_train_features)
                # print('res:{} , shape:{}'.format(output, output.shape))

                # 反向传播，获得最佳模型
                loss = loss_function(predict, Variable(cur_train_labels))
                print('epoch:{} , loss:{}'.format(epoch, loss))
                loss.backward()
                optimizer.step()
        torch.save(self.code_feature_model.state_dict(), 'code.pt')
