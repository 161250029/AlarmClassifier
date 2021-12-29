import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

from baseline.base2.service.CNN import CNN
import pandas as pd
from baseline.base2.service.Word2VecService import Word2VecSerice

# 8:2
def getData():
    data = pd.read_pickle('token.pkl')
    data = data.sample(frac=1, random_state=666)
    tokens = []
    labels = []
    for _, item in data.iterrows():
        tokens.append(item['token'])
        label = 1 if item['label'] == 'true' else 0
        labels.append(label)
    return tokens, torch.LongTensor(labels)

def split_data(tokens , labels):
    data_num = len(tokens)
    train_split = int(8 / 10 * data_num)
    train_data = tokens[:train_split]
    train_label = labels[:train_split]
    test_data = tokens[train_split:]
    test_label = labels[train_split:]
    return train_data , train_label , test_data , test_label

def vectorlize(word2vec , data):
    x = []
    for token_list in data:
        cur = []
        for token in token_list:
            cur.append(word2vec.get_vocab_index(token))
        x.append(cur)
    return x

def train():
    tokens , labels = getData()
    word2vec = Word2VecSerice(tokens , 'word.model')
    word2vec.train()

    train_data, train_labels, test_data, test_labels = split_data(tokens, labels)


    embedding = word2vec.get_embedding()
    vocab_size = word2vec.get_vocab_size()
    embedding_dim = word2vec.get_embedding_dim()
    model = CNN(vocab_size , embedding_dim , embedding)
    EPCHO = 10

    loss_function = torch.nn.BCELoss(
        weight=None,
        size_average=None,
        reduction="mean",
    )

    loss_function = torch.nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)

    for epcho in range(EPCHO):
        i = 0
        while i < len(train_data):
            train_data_feature = vectorlize(word2vec, train_data[i:i + 64])
            train_label = train_labels[i:i + 64]
            i += 64
            optimizer.zero_grad()
            output = model(train_data_feature)
            # 反向传播，获得最佳模型
            loss = loss_function(output, Variable(train_label))
            print('epcho:{} , loss:{}'.format(epcho , loss))
            _, predicted = torch.max(output, 1)
            loss.backward()
            optimizer.step()

    i = 0
    total_acc = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    while i < len(test_data):
        test_data_feature = vectorlize(word2vec, test_data[i:i + 64])
        test_label = test_labels[i:i + 64]
        i += 64
        output = model(test_data_feature)
        # 反向传播，获得最佳模型
        loss = loss_function(output, Variable(test_label))
        print('loss:{}'.format(loss))
        _, predicted = torch.max(output, 1)

        TP += ((predicted == 1) & (test_label.data == 1)).cpu().sum()
        # TN    predict 和 label 同时为0
        TN += ((predicted == 0) & (test_label.data == 0)).cpu().sum()
        # FN    predict 0 label 1
        FN += ((predicted == 0) & (test_label.data == 1)).cpu().sum()
        # FP    predict 1 label 0
        FP += ((predicted == 1) & (test_label.data == 0)).cpu().sum()

        print('predicted:{} , labels:{}'.format(predicted, test_label))
        total_acc += (predicted == test_label).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * recall * precision / (recall + precision)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("Testing results(Acc):", total_acc.item() / len(test_data))
    print('TP:{} , TN:{} , FP:{} , FN:{} , total_acc:{}'.format(TP , TN , FP , FN , total_acc))
    print("Precision:{} , Recall:{} , F1:{} , ACC:{}".format(precision , recall , f1 , acc))
    visualization(precision , recall , f1 , acc)

def visualization(precision , recall , f1 , acc):
    x = ['precision' , 'recall' , 'f1' , 'acc']
    y = [precision , recall , f1 , acc]
    fig, ax = plt.subplots()
    b = ax.bar(x, y)
    plt.bar(range(len(y)), y, color='rbg', tick_label=x)
    for a, b in zip(x, y):
        ax.text(a, b + 1, b, ha='center', va='bottom')

    plt.title('Numbers of Four eventtypes')
    plt.xlabel('Eventtype')
    plt.ylabel('Number')

    plt.savefig('./test.png')

if __name__ == '__main__':
    train()

