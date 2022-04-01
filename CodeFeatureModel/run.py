import torch
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np

from CodeFeatureModel.EmbeddingBuilder import EmbeddingBuilder
from CodeFeatureModel.SplitAST import SplitAST
from CodeFeatureModel.model import CodeFeatureModel
from DetectModel.CodeFeatureModelBuilder import CodeFeatureModelBuilder


def getData(data_path: str):
    data = pd.read_pickle(data_path)
    tokens = []
    labels = []
    for _, item in data.iterrows():
        tokens.append(item['ast'])
        label = 1 if item['label'] == 'true' else 0
        labels.append(label)
    return tokens, torch.LongTensor(labels)
if __name__ == '__main__':
    train_data, train_label = getData('/root/GY/Experiment/hadoop-common/3.1.4/ast.pkl')
    print('embedding building')
    embedding_builder = EmbeddingBuilder(train_data)
    embedding_builder.train_embedding()
    print('embedding end')
    vocab_model = Word2Vec.load('w2vModel.model')
    print('split building')
    split_ast = SplitAST(vocab_model , train_data)
    train_data = split_ast.vectorlize()
    print('split end')

    print('model building')
    embeddings = np.zeros((vocab_model.wv.syn0.shape[0] + 1, vocab_model.wv.syn0.shape[1]), dtype="float32")
    embeddings[:vocab_model.wv.syn0.shape[0]] = vocab_model.wv.syn0
    MAX_TOKENS = vocab_model.wv.syn0.shape[0]
    EMBEDDING_DIM = vocab_model.wv.syn0.shape[1]
    print('MAX_TOKENS:{} , EMBEDDING_DIM:{}'.format(MAX_TOKENS, EMBEDDING_DIM))
    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 2
    BATCH_SIZE = 64
    code_feature_model = CodeFeatureModelBuilder(train_data, train_label, EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE, embeddings)
    code_feature_model.train()
    print('model end')
    # print(torch.load('code.pt'))

    model = CodeFeatureModel(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE, embeddings)
    model.load_state_dict(torch.load('code.pt'))