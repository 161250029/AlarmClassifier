import os

import pandas as pd
import torch

import numpy as np

from baseline.base import Base
from baseline.base6.Word2VecService import Word2VecSerice
from baseline.base6.linear import Linear


class BaseLine6(Base):

    def __init__(self):
        super(BaseLine6, self).__init__()

        self.train_data_file_name = 'trainBestFeatures.pkl'
        self.test_data_file_name = 'testBestFeatures.pkl'

        self.vocab_model = None

        self.train_code_data = None

        self.test_code_data = None

        self.baseName = 'BaseLine6'

    def getData(self , data_path):
        data = pd.read_pickle(data_path)
        tokens = []
        labels = []
        for _, item in data.iterrows():
            astnn = item['astnn']
            tokens.append(astnn)
            label = 1 if item['label'] == 'true' else 0
            labels.append(label)
        return tokens, torch.LongTensor(labels)

    def getCode(self , data_path):
        data = pd.read_pickle(data_path)
        tokens = []
        for _, item in data.iterrows():
            code = item['code'].strip().replace("\n", " ").split(' ')
            code = list(filter(None, code))
            tokens.append(code)
        return tokens

    def train_vocab(self):
        train_dir = os.path.dirname(os.path.abspath(self.train_data_path))
        self.train_code_data = self.getCode(os.path.join(train_dir , 'ast.pkl'))
        test_dir = os.path.dirname(os.path.abspath(self.test_data_path))
        self.test_code_data = self.getCode(os.path.join(test_dir , 'ast.pkl'))
        total = self.train_code_data.copy()
        total.extend(self.test_code_data)
        self.vocab_model = Word2VecSerice(total, 'word.model')
        self.vocab_model.train()
        print(self.vocab_model.get_vocab_size())

    def vectorlize(self, batch_data):
        # 全部集合
        features = []
        for token_list in batch_data:
            cur = []
            for token in token_list:
                cur.append(self.vocab_model.get_vector(token))
            cur = np.array(cur)
            features.append(list(np.average(cur , axis=0)))
        return features

    def train(self , train_data, train_label, test_data, test_label):
        train_features = []
        test_features = []
        for i in range(len(train_data)):
            temp = train_data[i].copy()
            temp.extend(self.vectorlize([self.train_code_data[i]])[0])
            train_features.append(temp)

        for i in range(len(test_data)):
            temp = test_data[i].copy()
            temp.extend(self.vectorlize([self.test_code_data[i]])[0])
            test_features.append(temp)

        hidden_dim = len(train_features[0])
        output_size = 2

        model = Linear(hidden_dim , output_size)
        parameters = model.parameters()
        optimizer = torch.optim.Adadelta(parameters)
        loss_function = torch.nn.CrossEntropyLoss()

        EPOCHS = self.EPOCHS
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
    baselin6 = BaseLine6()
    baselin6.run()



