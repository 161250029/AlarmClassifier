import pandas as pd
import torch
from torch.autograd import Variable

from baseline.base1.LSTM import lstm
from baseline.base1.Word2VecService import Word2VecSerice
from baseline.base import Base
import numpy as np

class BaseLine1(Base):
    def __init__(self):
        super(BaseLine1, self).__init__()

        self.train_data_file_name = 'byteToken.pkl'
        self.test_data_file_name = 'byteToken.pkl'

        self.vocab_model = None
        self.baseName = 'BaseLine1'

    def getData(self , data_path):
        # Config.byteTokenStorePath
        data = pd.read_pickle(data_path)
        tokens = []
        labels = []
        for _, item in data.iterrows():
            tokens.append(item['code'].split(' '))
            label = 1 if item['label'] == 'true' else 0
            labels.append(label)
        return tokens, torch.LongTensor(labels)

    def train_vocab(self):
        self.vocab_model = Word2VecSerice(self.total_data, 'word.model')
        self.vocab_model.train()

    def vectorlize(self , batch_data):
        # 全部集合
        # 选择当前批次的最大len
        lens = [len(i) for i in batch_data]
        max_lens = max(lens)
        features = []
        for token_list in batch_data:
            cur = []
            for token in token_list:
                cur.append(self.vocab_model.get_vector(token))
            if max_lens - len(token_list):
                cur.extend(list(np.zeros((max_lens - len(token_list), 128), dtype=float)))
            features.append(cur)
        features = np.array(features)
        features.reshape(-1, 128)
        features = torch.from_numpy(features)
        features = features.view(-1, max_lens, 128)
        # print(features.shape)
        return features.to(torch.float32)

    def train(self ,train_data, train_label, test_data, test_label):
        # 维度太高，无法一次性给所有样本进行编码
        # train_features = vectorlize(tokens, train_data)
        # test_features = vectorlize(tokens, test_data)

        EPOCHS = 15

        input_size = 128
        # 4
        hidden_size = 200
        output_size = 2
        num_layer = 1
        model = lstm(input_size, hidden_size, output_size, num_layer)
        parameters = model.parameters()
        optimizer = torch.optim.Adadelta(parameters)
        loss_function = torch.nn.CrossEntropyLoss()

        batch_size = 32

        for epcho in range(EPOCHS):
            i = 0
            while i < len(train_data):
                cur_train_features = self.vectorlize(train_data[i:i + batch_size])
                cur_train_labels = train_label[i:i + batch_size]

                optimizer.zero_grad()

                output = model(cur_train_features)

                # 反向传播，获得最佳模型
                loss = loss_function(output, Variable(cur_train_labels))
                print('epcho:{} , loss:{}'.format(epcho, loss))
                loss.backward()
                optimizer.step()
                i += batch_size

        i = 0
        while i < len(test_data):
            cur_test_features = self.vectorlize(test_data[i: i + batch_size])
            cur_test_labels = test_label[i: i + batch_size]
            output = model(cur_test_features)
            cur_test_labels = torch.LongTensor(cur_test_labels)
            _, predicted = torch.max(output, 1)
            self.statistic(predicted , cur_test_labels , output)
            i += batch_size

if __name__ == '__main__':
    baseline1 = BaseLine1()
    baseline1.run()



