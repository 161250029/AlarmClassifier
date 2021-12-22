import pandas as pd
import torch
from torch.autograd import Variable

from baseline.Config import Config
from baseline.base1.LSTM import lstm
from baseline.base1.Word2VecService import Word2VecSerice
import numpy as np

def getData():
    data = pd.read_pickle(Config.byteTokenStorePath)
    data = data.sample(frac=1, random_state=666)
    tokens = []
    labels = []
    for _, item in data.iterrows():
        tokens.append(item['token'])
        label = 1 if item['label'] == 'true' else 0
        labels.append(label)
    return tokens, torch.LongTensor(labels)

def vectorlize(total_tokens , data):
    # 全部集合
    model = Word2VecSerice(total_tokens, 'word.model')
    model.train()
    lens = [len(i) for i in data]
    max_lens = max(lens)
    features = []
    for token_list in data:
        cur = []
        for token in token_list:
            cur.append(model.get_vocab(token))
        if max_lens - len(token_list):
            cur.append(np.zeros([max_lens - len(token_list) , 128] , dtype = float))
        features.append(cur)
    features.reshape(-1 , 128)
    features = torch.from_numpy(features)
    print(features.shape)
    features = features.view(-1 , max_lens , 128)
    print(features.shape)
    return features

def split_data(tokens , labels):
    data_num = len(tokens)
    train_split = int(8 / 10 * data_num)
    train_data = tokens[:train_split]
    train_label = labels[:train_split]
    test_data = tokens[train_split:]
    test_label = labels[train_split:]
    return train_data , train_label , test_data , test_label

def train():
    tokens, labels = getData()
    train_data, train_label, test_data, test_label = split_data(tokens, labels)

    train_features = vectorlize(tokens, train_data)
    test_features = vectorlize(tokens, test_data)


    input_size = 128
    hidden_size = 4
    output_size = 2
    num_layer = 1
    model = lstm(input_size , hidden_size , output_size , num_layer)
    parameters = model.parameters()
    optimizer = torch.optim.Adadelta(parameters)
    loss_function = torch.nn.CrossEntropyLoss()


    batch_size = 64
    i = 0
    while i < train_features.shape(0):
        cur_train_features = train_features[i:i+batch_size]
        cur_train_labels = train_label[i:i+batch_size]

        output = model(cur_train_features)

        # 反向传播，获得最佳模型
        loss = loss_function(output, Variable(cur_train_labels))
        print('i:{} , loss:{}'.format(i , loss))
        loss.backward()
        optimizer.step()
        i += batch_size

    i = 0
    total_acc = 0
    total = 0
    while i < test_features.shape(0):
        cur_test_features = test_features[i : i + batch_size]
        cur_test_labels = test_label[i : i + batch_size]

        output = model(cur_test_features)

        loss = loss_function(output, Variable(cur_test_labels))
        print(loss)

        _, predicted = torch.max(output, 1)
        print('predicted:{} , dev_labels:{}'.format(predicted, cur_test_labels))
        total_acc += (predicted == cur_test_labels).sum()
        total += len(cur_test_labels)
    print("Testing results(Acc):", total_acc.item() / total)



