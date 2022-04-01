import os

import torch

import pandas as pd
import numpy as np

from CodeFeatureModel.EmbeddingBuilder import EmbeddingBuilder
from CodeFeatureModel.SplitAST import SplitAST
from CodeFeatureModel.model import CodeFeatureModel
from DetectModel.CodeFeatureModelBuilder import CodeFeatureModelBuilder

from gensim.models.word2vec import Word2Vec


class Application:

    def __init__(self):
        self.root = '/root/GY/Experiment/'
        self.projects = []


    def init_projects(self):
        if os.path.exists(self.root):
            files = os.listdir(self.root)
            for file in files:
                m = os.path.join(self.root , file)
                if os.path.isdir(m):
                    self.projects.append(file)

    def get_versions(self, project: str) -> list:
        versions = []
        project_dir_path = os.path.join(self.root, project)
        if os.path.exists(project_dir_path):
            files = os.listdir(project_dir_path)
            for file in files:
                m = os.path.join(project_dir_path, file)
                if os.path.isdir(m) and not file.__contains__('BaseLine'):
                    versions.append(file)
        return versions

    def getData(self,data_path: str):
        data = pd.read_pickle(data_path)
        tokens = []
        labels = []
        for _, item in data.iterrows():
            tokens.append(item['ast'])
            label = 1 if item['label'] == 'true' else 0
            labels.append(label)
        return tokens, torch.LongTensor(labels)

    def run(self):
        train_data = []
        train_label = []
        self.init_projects()
        for project in self.projects:
            versions = self.get_versions(project)
            for version in versions:
                data_path = os.path.join(self.root, project, version, 'ast.pkl')
                cur_train_data, cur_train_label = self.getData(data_path)
                train_data.extend(cur_train_data)
                train_label.append(cur_train_label)
        train_label = torch.cat(train_label)
        print('embedding building')
        embedding_builder = EmbeddingBuilder(train_data)
        embedding_builder.train_embedding()
        print('embedding end')
        vocab_model = Word2Vec.load('w2vModel.model')
        print('split building')
        split_ast = SplitAST(vocab_model, train_data)
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
        code_feature_model = CodeFeatureModelBuilder(train_data, train_label, EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1,
                                                     ENCODE_DIM, LABELS, BATCH_SIZE, embeddings)
        code_feature_model.train()
        print('model end')

        model = CodeFeatureModel(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE, embeddings)
        model.load_state_dict(torch.load('code.pt'))

        print(torch.load('code.pt'))

if __name__ == '__main__':
    application = Application()
    application.run()
