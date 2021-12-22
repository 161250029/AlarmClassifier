from gensim.models import Word2Vec

class Word2VecSerice:

    # text: 二维数组
    def __init__(self , texts , model_path):
        self.model = None
        self.texts = texts
        self.model_path = model_path


    def train(self):
        # skip_gram
        self.model = Word2Vec(sentences=self.texts, size=128, window=5, min_count=1, workers=4 , sg=1)
        self.model.save(self.model_path)

    def load(self):
        self.model = Word2Vec.load(self.model_path)
        return self.model

    def get_vector(self , vocab):
        if self.model == None:
            self.load()
        return self.model.wv[vocab]

    def get_vocab_index(self , token):
        if self.model == None:
            self.load()
        return self.model.wv.index2word.index(token)

    def get_embedding(self):
        if self.model == None:
            self.load()
        return self.model.wv.syn0

    def get_vocab_size(self):
        if self.model == None:
            self.load()
        return self.model.wv.syn0.shape[0]

    def get_embedding_dim(self):
        if self.model == None:
            self.load()
        return self.model.wv.syn0.shape[1]


if __name__ == '__main__':
    service = Word2VecSerice([['i' , 'love' , 'u'] , ['i' , 'like' , 'him']] , 'word.model')
    service.train()
    print(service.model.wv.index2word[0])
    print(service.model.wv.syn0.shape)
    print(service.get_vector('i'))