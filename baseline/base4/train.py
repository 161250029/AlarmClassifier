import os

import torch
import pandas as pd
from gensim.models.word2vec import Word2Vec
import numpy as np
from torch.autograd import Variable

from baseline.base import Base
from baseline.base4.model import BatchProgramClassifier

from baseline.base4.utils import get_blocks_v1 as func

from baseline.base4.utils import get_sequence

class BaseLine4(Base):

    def __init__(self):
        super(BaseLine4, self).__init__()

        self.train_data_file_name = 'ast.pkl'
        self.test_data_file_name = 'ast.pkl'

        self.vocab_model = None

        self.baseName = 'BaseLine4'

    def getData(self ,data_path):
        data = pd.read_pickle(data_path)
        tokens = []
        labels = []
        for _, item in data.iterrows():
            tokens.append(item['code'])
            label = 1 if item['label'] == 'true' else 0
            labels.append(label)
        return tokens, torch.LongTensor(labels)

    # data的形式应该为语法树节点
    def vectorlize(self, batch_data):
        vocab = self.vocab_model.wv.vocab
        max_token = self.vocab_model.wv.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        features = []
        for data in batch_data:
            features.append(trans2seq(data))
        return features

    def train_vocab(self):
        def trans_to_sequences(ast):
            sequence = []
            # 获得ast的token序列,ast即FileAST根节点
            get_sequence(ast, sequence)
            return sequence
        corpus = []
        for node in self.train_data:
            corpus.append(trans_to_sequences(node))
        w2v = Word2Vec(corpus, size=128, workers=16, sg=1, min_count=3)
        self.vocab_model = w2v


    def train(self , train_data, train_label, test_data, test_label):
        word2vec = self.vocab_model.wv
        embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
        embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

        MAX_TOKENS = word2vec.syn0.shape[0]
        EMBEDDING_DIM = word2vec.syn0.shape[1]
        print('MAX_TOKENS:{} , EMBEDDING_DIM:{}'.format(MAX_TOKENS, EMBEDDING_DIM))
        HIDDEN_DIM = 100
        ENCODE_DIM = 128
        LABELS = 2
        EPOCHS = 15
        BATCH_SIZE = 64
        USE_GPU = False

        m = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                                   USE_GPU, embeddings)

        parameters = m.parameters()
        optimizer = torch.optim.Adamax(parameters)
        loss_function = torch.nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            i = 0
            while i < len(train_data):
                cur_train_features = self.vectorlize(train_data[i:i + BATCH_SIZE])
                cur_train_labels = train_label[i:i + BATCH_SIZE]
                i += BATCH_SIZE
                optimizer.zero_grad()
                # 不足批次大小的要重新设置batchsize
                m.batch_size = len(cur_train_labels)
                # 同时需要重置隐藏层
                m.hidden = m.init_hidden()
                output, predict = m(cur_train_features)
                # print('res:{} , shape:{}'.format(output, output.shape))

                # 反向传播，获得最佳模型
                loss = loss_function(predict, Variable(cur_train_labels))
                print('epoch:{} , loss:{}'.format(epoch, loss))
                loss.backward()
                optimizer.step()

        i = 0
        astnnFeatures = []
        while i < len(train_data):
            cur_train_features = self.vectorlize(train_data[i:i + BATCH_SIZE])
            cur_train_labels = train_label[i:i + BATCH_SIZE]
            i += BATCH_SIZE
            # 不足批次大小的要重新设置batchsize
            m.batch_size = len(cur_train_labels)
            # 同时需要重置隐藏层
            m.hidden = m.init_hidden()
            output, predict = m(cur_train_features)
            astnnFeatures.append(output)

        astnnFeatures = torch.cat(astnnFeatures)
        # tensor转list
        train_data_frame = pd.read_pickle(self.train_data_path)
        train_data_frame['astnn'] = astnnFeatures.tolist()
        train_data_frame.to_pickle(os.path.join(os.path.split(self.train_data_path)[0] , 'trainBestFeatures.pkl'))

        astnnFeatures = []
        i = 0
        while i < len(test_data):
            cur_test_features = self.vectorlize(test_data[i: i + BATCH_SIZE])
            cur_test_labels = test_label[i: i + BATCH_SIZE]
            i += BATCH_SIZE
            # 不足批次大小的要重新设置batchsize
            m.batch_size = len(cur_test_labels)
            # 同时需要重置隐藏层
            m.hidden = m.init_hidden()
            output, predict = m(cur_test_features)
            astnnFeatures.append(output)
            # print('res:{} , shape:{}'.format(output, output.shape))
            _, predicted = torch.max(predict, 1)
            print('predicted:{} , test_labels:{}'.format(predicted, cur_test_labels))
            self.statistic(predicted , cur_test_labels , output)

        astnnFeatures = torch.cat(astnnFeatures)
        # tensor转list
        test_data_frame = pd.read_pickle(self.test_data_path)
        test_data_frame['astnn'] = astnnFeatures.tolist()
        test_data_frame.to_pickle(os.path.join(os.path.split(self.test_data_path)[0] , 'testBestFeatures.pkl'))


if __name__ == '__main__':
    baseline4 = BaseLine4()
    baseline4.run()