import torch

from CodeFeatureModel.SplitAST import SplitAST
from CodeFeatureModel.model import CodeFeatureModel

from gensim.models.word2vec import Word2Vec

import numpy as np

from service.ModelConfig import ModelConfig

import pandas as pd


# 检测入口
class Detect:
    def __init__(self, test_data: list):
        self.test_data = test_data
        self.vocab_model = Word2Vec.load(ModelConfig.vocab_model_path)

        self.embeddings = np.zeros((self.vocab_model.wv.syn0.shape[0] + 1, self.vocab_model.wv.syn0.shape[1]), dtype="float32")
        self.embeddings[:self.vocab_model.wv.syn0.shape[0]] = self.vocab_model.wv.syn0
        self.MAX_TOKENS = self.vocab_model.wv.syn0.shape[0]
        self.EMBEDDING_DIM = self.vocab_model.wv.syn0.shape[1]

        self.HIDDEN_DIM = 100
        self.ENCODE_DIM = 128
        self.LABELS = 2
        self.BATCH_SIZE = 64

        self.code_feature_model = CodeFeatureModel(self.EMBEDDING_DIM, self.HIDDEN_DIM, self.MAX_TOKENS + 1, self.ENCODE_DIM, self.LABELS, self.BATCH_SIZE, self.embeddings)
        self.code_feature_model.load_state_dict(torch.load(ModelConfig.model_paramenter_path))

    def detect(self):
        split_ast = SplitAST(self.vocab_model, self.test_data)
        test_data = split_ast.vectorlize()
        i = 0
        code_vector = []
        while i < len(test_data):
            cur_test_data = test_data[i:i + self.BATCH_SIZE]
            i += self.BATCH_SIZE
            # 不足批次大小的要重新设置batchsize
            self.code_feature_model.batch_size = len(cur_test_data)
            # 同时需要重置隐藏层
            self.code_feature_model.hidden = self.code_feature_model.init_hidden()
            output, predict = self.code_feature_model(cur_test_data)
            code_vector.append(output)
            _, predicted = torch.max(predict, 1)
            print('predicted:{}'.format(predicted))
        code_vector = torch.cat(code_vector)
        print(code_vector)


def getData(data_path: str):
    data = pd.read_pickle(data_path)
    tokens = []
    for _, item in data.iterrows():
        tokens.append(item['ast'])
    return tokens

if __name__ == '__main__':
    train_data = getData('/root/GY/ast.pkl')
    detect = Detect(train_data)
    detect.detect()


