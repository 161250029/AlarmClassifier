import pandas as pd

from baseline.base5.OneHotEncode import OneHotEncode


class ExtractReportFeature:
    def __init__(self):
        self.one_hot_columns = ['type',  'priority' , 'catogray' , 'rank']
        self.normal_columns = ['start']
        self.encode = OneHotEncode()

    def extract(self , dataPath1 , dataPath2):
        dataframe1 = pd.read_pickle(dataPath1)
        dataframe2 = pd.read_pickle(dataPath2)
        res = []
        vectors = []
        for one_hot_column in self.one_hot_columns:
            features = dataframe1[one_hot_column].tolist()
            features.extend(dataframe2[one_hot_column].tolist())
            features = [[i] for i in features]
            vector = self.encode.transform(features)
            vectors.append(vector)

        for normal_column in self.normal_columns:
            features = dataframe1[normal_column].tolist()
            features.extend(dataframe2[normal_column].tolist())
            features = list(map(int,features))
            vector = self.normalization(features)
            vectors.append(vector)

        for i in range(len(vectors[0])):
            vector = []
            for j in range(len(vectors)):
                vector.extend(vectors[j][i])
            res.append(vector)
        return res


    def normalization(self , data):
        min_data = min(data)
        max_data = max(data)
        _range = max_data - min_data
        return [[(x - min_data) / _range] for x in data]

if __name__ == '__main__':
    ExtractReportFeature().extract('../../prepareData/service/data/program.pkl')