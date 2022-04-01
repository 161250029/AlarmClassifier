
import torch
from torch.autograd import Variable

from baseline.base import Base
from baseline.base2.CNN import CNN
import pandas as pd
from baseline.base2.Word2VecService import Word2VecSerice


# 8:2
class BaseLine2(Base):

    def __init__(self):
        super(BaseLine2, self).__init__()

        self.train_data_file_name = 'fiveLines.pkl'
        self.test_data_file_name = 'fiveLines.pkl'

        self.vocab_model = None

        self.baseName = 'BaseLine2'

    def getData(self ,data_path):
        data = pd.read_pickle(data_path)
        tokens = []
        labels = []
        for _, item in data.iterrows():
            tokens.append(item['code'].strip().replace("/n", " ").split(" "))
            label = 1 if item['label'] == 'true' else 0
            labels.append(label)
        return tokens, torch.LongTensor(labels)

    def train_vocab(self):
        self.vocab_model = Word2VecSerice(self.total_data, 'word.model')
        self.vocab_model.train()

    def vectorlize(self ,batch_data):
        x = []
        for token_list in batch_data:
            cur = []
            for token in token_list:
                cur.append(self.vocab_model.get_vocab_index(token))
            x.append(cur)
        return x

    def train(self , train_data, train_label, test_data, test_label):
        embedding = self.vocab_model.get_embedding()
        vocab_size = self.vocab_model.get_vocab_size()
        embedding_dim = self.vocab_model.get_embedding_dim()
        model = CNN(vocab_size, embedding_dim, embedding)
        EPCHO = self.EPOCHS

        loss_function = torch.nn.CrossEntropyLoss()
        parameters = model.parameters()
        optimizer = torch.optim.Adamax(parameters)

        for epcho in range(EPCHO):
            i = 0
            while i < len(train_data):
                cur_train_features = self.vectorlize(train_data[i:i + 64])
                cur_train_labels = train_label[i:i + 64]
                i += 64
                optimizer.zero_grad()
                output = model(cur_train_features)
                # 反向传播，获得最佳模型
                loss = loss_function(output, Variable(cur_train_labels))
                print('epcho:{} , loss:{}'.format(epcho, loss))
                _, predicted = torch.max(output, 1)
                loss.backward()
                optimizer.step()

        i = 0
        while i < len(test_data):
            cur_test_features = self.vectorlize(test_data[i:i + 64])
            cur_test_labels = test_label[i:i + 64]
            i += 64
            output = model(cur_test_features)
            _, predicted = torch.max(output, 1)
            self.statistic(predicted , cur_test_labels , output)


if __name__ == '__main__':
    baseLine2 = BaseLine2()
    baseLine2.run()