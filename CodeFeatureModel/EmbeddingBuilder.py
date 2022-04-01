from gensim.models.word2vec import Word2Vec

from CodeFeatureModel.utils import get_sequence


class EmbeddingBuilder:

    def __init__(self , train_data: list):
        self.train_data = train_data
        self.vocab_model = None

    def train_embedding(self):
        def trans_to_sequences(ast):
            sequence = []
            # 获得ast的token序列,ast即FileAST根节点
            get_sequence(ast, sequence)
            return sequence
        corpus = []
        for node in self.train_data:
            corpus.append(trans_to_sequences(node))
        word2vec = Word2Vec(corpus, size=128, workers=16, sg=1, min_count=3)
        word2vec.save('w2vModel.model')
