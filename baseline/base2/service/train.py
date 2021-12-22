import torch
from torch.autograd import Variable

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

def train():
    tokens , labels = getData()
    word2vec = Word2VecSerice(tokens , 'word.model')
    word2vec.train()
    x = []
    for token_list in tokens:
        cur = []
        for token in token_list:
            cur.append(word2vec.get_vocab_index(token))
        x.append(cur)
    embedding = word2vec.get_embedding()
    vocab_size = word2vec.get_vocab_size()
    embedding_dim = word2vec.get_embedding_dim()
    model = CNN(vocab_size , embedding_dim , embedding)

    i = 0
    loss_function = torch.nn.BCELoss(
        weight=None,
        size_average=None,
        reduction="mean",
    )
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    total_acc = 0

    output = model(x)
    # 反向传播，获得最佳模型
    loss = loss_function(output, Variable(labels))
    print('loss:{}'.format(loss))
    _, predicted = torch.max(output, 1)
    print('predicted:{} , labels:{}'.format(predicted, labels))
    total_acc += (predicted == labels).sum()
    # while i < len(x):
    #     train_data = x[i:i + 64]
    #     label = labels[i:i + 64]
    #     i+= 64
    #     optimizer.zero_grad()
    #     output = model(train_data)
    #     # 反向传播，获得最佳模型
    #     loss = loss_function(output, Variable(label))
    #     print('loss:{}'.format(loss))
    #     _, predicted = torch.max(output, 1)
    #     print('predicted:{} , labels:{}'.format(predicted, label))
    #     total_acc += (predicted == label).sum()
    #     loss.backward()
    #     optimizer.step()

    print(total_acc)

if __name__ == '__main__':
    train()

