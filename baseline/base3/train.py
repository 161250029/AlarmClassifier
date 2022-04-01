import pandas as pd
import torch

from baseline.base import Base
from baseline.base3.BowModel import BowModel
from baseline.base3.LSTM import lstm


class BaseLine3(Base):

    def __init__(self):
        super(BaseLine3, self).__init__()

        self.train_data_file_name = 'sliceToken.pkl'
        self.test_data_file_name = 'sliceToken.pkl'

        self.vocab_model = None

        self.baseName = 'BaseLine3'

    def getData(self , data_path):
        data = pd.read_pickle(data_path)
        tokens = []
        labels = []
        for _, item in data.iterrows():
            tokens.append(item['code'][:-1].split(','))
            label = 1 if item['label'] == 'true' else 0
            labels.append(label)
        return tokens, torch.LongTensor(labels)

    def vectorlize(self, batch_data):
        features = self.vocab_model.transform(batch_data)
        # print(features.shape)
        return features

    def train_vocab(self):
        self.vocab_model = BowModel(self.total_data)
        self.vocab_model.fit()

    def train(self , train_data, train_label, test_data, test_label):
        train_features = self.vectorlize(train_data)
        test_features = self.vectorlize(test_data)

        input_size = train_features.shape[1]
        hidden_size = 200
        output_size = 2
        num_layer = 1

        train_features = torch.from_numpy(train_features)
        train_features = train_features.view(-1, 1, input_size)
        train_features = train_features.to(torch.float32)

        test_features = torch.from_numpy(test_features)
        test_features = test_features.view(-1, 1, input_size)
        test_features = test_features.to(torch.float32)

        model = lstm(input_size, hidden_size, output_size, num_layer)
        parameters = model.parameters()
        optimizer = torch.optim.Adadelta(parameters)
        loss_function = torch.nn.CrossEntropyLoss()

        EPOCHS = self.EPOCHS
        batch_size = 64
        for epcho in range(EPOCHS):
            i = 0
            while i < train_features.size(0):
                cur_train_features = train_features[i:i + batch_size]
                cur_train_labels = train_label[i:i + batch_size]

                optimizer.zero_grad()

                output = model(cur_train_features)

                # 反向传播，获得最佳模型
                loss = loss_function(output, cur_train_labels)
                print('epcho:{} , loss:{}'.format(epcho, loss))
                loss.backward()
                optimizer.step()
                i += batch_size

        i = 0
        while i < test_features.size(0):
            cur_test_features = test_features[i: i + batch_size]
            cur_test_labels = test_label[i: i + batch_size]
            output = model(cur_test_features)
            cur_test_labels = torch.LongTensor(cur_test_labels)
            _, predicted = torch.max(output, 1)
            self.statistic(predicted , cur_test_labels , output)
            i += batch_size
if __name__ == '__main__':
    baseline3 = BaseLine3()
    baseline3.re_run()



