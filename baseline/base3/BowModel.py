
# CountVectorizer：对语料库中出现的词汇进行词频统计，相当于词袋模型。
# 操作方式：将语料库当中出现的词汇作为特征，将词汇在当前文档中出现的频率（次数）作为特征值。
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class BowModel:
    # docs:token序列的数组
    # 需要将数组转换成字符串
    def __init__(self , docs):
        self.docs = np.array([' '.join(doc) for doc in docs])
        self.model = CountVectorizer()

    def fit(self):
        self.model.fit(self.docs)

    def transform(self , tokens):
        tokens = np.array([' '.join(token) for token in tokens])
        bag = self.model.transform(tokens)
        return bag.toarray()


if __name__ == '__main__':
    docs = [['public' , 'static' , 'void'] , ['public' , 'final' , 'like'] , ['String' , 'val1']]
    model = BowModel(docs)
    tokens = [['static' , 'final'] , ['public']]
    model.fit()
    print(model.transform(tokens))
    print(model.transform(tokens).shape[1])
