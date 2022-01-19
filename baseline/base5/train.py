import math

import pandas as pd
import torch

from baseline.base import Base
from baseline.base5.extractReportFeature import ExtractReportFeature
from baseline.base5.gru import GRU

class BaseLine5(Base):

    def __init__(self):
        super(BaseLine5, self).__init__()

        self.train_data_file_name = 'trainBestFeatures.pkl'
        self.test_data_file_name = 'testBestFeatures.pkl'

        self.vocab_model = None

        self.baseName = 'BaseLine5'

    def getData(self , data_path):
        data = pd.read_pickle(data_path)
        tokens = []
        labels = []
        index = 0
        for _, item in data.iterrows():
            astnn = item['astnn']
            tokens.append(astnn)
            label = 1 if item['label'] == 'true' else 0
            labels.append(label)
            index += 1
        return tokens, torch.LongTensor(labels)

    def vectorlize(self, batch_data):
        pass

    def train_vocab(self):
        pass

    def train(self , train_data, train_label, test_data, test_label):
        metrics = ExtractReportFeature().extract(self.train_data_path , self.test_data_path)
        train_features = []
        test_features = []
        for i in range(len(train_data)):
            temp = train_data[i].copy()
            temp.extend(metrics[i])
            train_features.append(temp)

        for i in range(len(test_data)):
            temp = test_data[i].copy()
            temp.extend(metrics[i + len(train_data)])
            test_features.append(temp)

        hidden_dim = 200
        encode_dim = len(train_features[0])
        output_size = 2

        model = GRU(hidden_dim , encode_dim , output_size)
        parameters = model.parameters()
        optimizer = torch.optim.Adadelta(parameters)
        loss_function = torch.nn.CrossEntropyLoss()

        EPOCHS = 15
        batch_size = 64
        for epcho in range(EPOCHS):
            i = 0
            while i < len(train_features):
                cur_train_features = train_features[i:i + batch_size]
                cur_train_labels = train_label[i:i + batch_size]

                optimizer.zero_grad()

                output = model(cur_train_features)
                loss = loss_function(output, cur_train_labels)
                print('epcho:{} , loss:{}'.format(epcho, loss))
                loss.backward()
                optimizer.step()
                i += batch_size

        i = 0
        while i < len(test_features):
            cur_test_features = test_features[i: i + batch_size]
            cur_test_labels = test_label[i: i + batch_size]

            output = model(cur_test_features)

            cur_test_labels = torch.LongTensor(cur_test_labels)

            _, predicted = torch.max(output, 1)
            self.statistic(predicted , cur_test_labels , output)
            i += batch_size
if __name__ == '__main__':
    baselin5 = BaseLine5()
    baselin5.run()



