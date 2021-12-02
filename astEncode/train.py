import torch
import pandas as pd
from gensim.models.word2vec import Word2Vec
import numpy as np
from torch.autograd import Variable

import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
if p not in sys.path:
    sys.path.append(p)
from astEncode import BatchProgramClassifier

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item['code'])
        label = 1 if item['label'] == 'true' else 0
        labels.append(label)
    return data, torch.LongTensor(labels)

def train():
    word2vec = Word2Vec.load("../prepareData/service/train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    print('MAX_TOKENS:{} , EMBEDDING_DIM:{}'.format(MAX_TOKENS , EMBEDDING_DIM))
    i = 0
    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 2
    # EPOCHS = 15
    BATCH_SIZE = 64
    USE_GPU = False
    train_data = pd.read_pickle('../prepareData/service/train/blocks.pkl')

    m = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                                   USE_GPU, embeddings)
    astnnFeatures = []
    while i < len(train_data):
        print('{}-{} data start to train.'.format(i + 1, i + BATCH_SIZE))
        batch = get_batch(train_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        train_inputs, train_labels = batch
        # 不足批次大小的要重新设置batchsize
        m.batch_size = len(train_labels)
        # 同时需要重置隐藏层
        m.hidden = m.init_hidden()
        output , predict = m(train_inputs)
        astnnFeatures.append(output)
        print('res:{} , shape:{}'.format(output , output.shape))
    astnnFeatures = torch.cat(astnnFeatures)
    # tensor转list
    train_data['astnn'] = astnnFeatures.tolist()
    train_data.to_pickle('../prepareData/service/train/features.pkl')

def trainWithLoss():
    word2vec = Word2Vec.load("../prepareData/service/train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    print('MAX_TOKENS:{} , EMBEDDING_DIM:{}'.format(MAX_TOKENS, EMBEDDING_DIM))
    i = 0
    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 2
    # EPOCHS = 15
    BATCH_SIZE = 64
    USE_GPU = False
    train_data = pd.read_pickle('../prepareData/service/train/blocks.pkl')

    m = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                               USE_GPU, embeddings)

    parameters = m.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()

    bestmodel = m

    while i < len(train_data):
        print('{}-{} data start to train.'.format(i + 1, i + BATCH_SIZE))
        batch = get_batch(train_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        train_inputs, train_labels = batch
        optimizer.zero_grad()
        # 不足批次大小的要重新设置batchsize
        m.batch_size = len(train_labels)
        # 同时需要重置隐藏层
        m.hidden = m.init_hidden()
        output, predict = m(train_inputs)
        print('res:{} , shape:{}'.format(output, output.shape))

        # 反向传播，获得最佳模型
        loss = loss_function(output, Variable(train_labels))
        loss.backward()
        optimizer.step()

    m = bestmodel
    dev_data = pd.read_pickle('../prepareData/service/dev/blocks.pkl')
    astnnFeatures = []
    i = 0
    total_acc = 0
    total = 0
    total_loss = 0
    while i < len(dev_data):
        batch = get_batch(dev_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        dev_inputs,dev_labels = batch
        # 不足批次大小的要重新设置batchsize
        m.batch_size = len(dev_labels)
        # 同时需要重置隐藏层
        m.hidden = m.init_hidden()
        output , predict = m(dev_inputs)
        astnnFeatures.append(output)
        print('res:{} , shape:{}'.format(output , output.shape))
        loss = loss_function(output, Variable(dev_labels))
        print(loss)

        _, predicted = torch.max(predict, 1)
        print(predicted)
        print(dev_labels)
        total_acc += (predicted == dev_labels).sum()
        total += len(dev_labels)
        total_loss += loss.item() * len(dev_inputs)
    print("Testing results(Acc):", total_acc.item() / total)
    astnnFeatures = torch.cat(astnnFeatures)
    # tensor转list
    dev_data['astnn'] = astnnFeatures.tolist()
    dev_data.to_pickle('../prepareData/service/dev/bestFeatures.pkl')

    test_data = pd.read_pickle('../prepareData/service/test/blocks.pkl')
    astnnFeatures = []
    i = 0
    total_acc = 0
    total = 0
    total_loss = 0
    while i < len(test_data):
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels = batch
        # 不足批次大小的要重新设置batchsize
        m.batch_size = len(test_labels)
        # 同时需要重置隐藏层
        m.hidden = m.init_hidden()
        output, predict = m(test_inputs)
        astnnFeatures.append(output)
        print('res:{} , shape:{}'.format(output, output.shape))
        loss = loss_function(output, Variable(test_labels))
        print(loss)

        _, predicted = torch.max(predict, 1)
        print('predicted:{} , test_labels:{}'.format(predicted , test_labels))
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print("Testing results(Acc):", total_acc.item() / total)
    astnnFeatures = torch.cat(astnnFeatures)
    # tensor转list
    test_data['astnn'] = astnnFeatures.tolist()
    test_data.to_pickle('../prepareData/service/test/bestFeatures.pkl')

if __name__ == '__main__':
    trainWithLoss()
