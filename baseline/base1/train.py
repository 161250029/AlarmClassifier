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
        tokens.append(item['code'].split(' '))
        label = 1 if item['label'] == 'true' else 0
        labels.append(label)
    return tokens, torch.LongTensor(labels)

def vectorlize(total_tokens , data):
    # 全部集合
    model = Word2VecSerice(total_tokens, 'word.model')
    model.train()
    lens = [len(i) for i in total_tokens]
    max_lens = max(lens)
    features = []
    for token_list in data:
        cur = []
        for token in token_list:
            cur.append(model.get_vector(token))
        if max_lens - len(token_list):
            cur.extend(list(np.zeros((max_lens - len(token_list) , 128) , dtype = float)))
        features.append(cur)
    features = np.array(features)
    print(features.shape)
    features.reshape(-1 , 128)
    features = torch.from_numpy(features)
    print(features.shape)
    features = features.view(-1 , max_lens , 128)
    print(features.shape)
    return features.to(torch.float32)

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

    # 维度太高，无法一次性给所有样本进行编码
    # train_features = vectorlize(tokens, train_data)
    # test_features = vectorlize(tokens, test_data)

    EPOCHS = 10

    input_size = 128
    # 4
    hidden_size = 200
    output_size = 2
    num_layer = 1
    model = lstm(input_size , hidden_size , output_size , num_layer)
    parameters = model.parameters()
    optimizer = torch.optim.Adadelta(parameters)
    loss_function = torch.nn.CrossEntropyLoss()


    batch_size = 64

    for epcho in range(EPOCHS):
        i = 0
        while i < len(train_data):
            cur_train_features = vectorlize(tokens, train_data[i:i + batch_size])
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
    total_acc = 0
    total = 0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    while i < len(test_data):
        cur_test_features = vectorlize(tokens , test_data[i : i + batch_size])
        cur_test_labels = test_label[i : i + batch_size]

        output = model(cur_test_features)

        loss = loss_function(output, Variable(cur_test_labels))
        print(loss)

        cur_test_labels = torch.LongTensor(cur_test_labels)


        _, predicted = torch.max(output, 1)

        TP += ((predicted == 1) & (cur_test_labels.data == 1)).cpu().sum()
        # TN    predict 和 label 同时为0
        TN += ((predicted == 0) & (cur_test_labels.data == 0)).cpu().sum()
        # FN    predict 0 label 1
        FN += ((predicted == 0) & (cur_test_labels.data == 1)).cpu().sum()
        # FP    predict 1 label 0
        FP += ((predicted == 1) & (cur_test_labels.data == 0)).cpu().sum()

        print('predicted:{} , dev_labels:{}'.format(predicted, cur_test_labels))
        total_acc += (predicted == cur_test_labels).sum()
        total += len(cur_test_labels)
        i += batch_size

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * recall * precision / (recall + precision)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("Testing results(Acc):", total_acc.item() / total)
    print('TP:{} , TN:{} , FP:{} , FN:{} , total_acc{}'.format(TP , TN , FP , FN , total_acc))
    print("Precision:{} , Recall:{} , F1:{} , ACC:{}".format(precision , recall , f1 , acc))

train()



