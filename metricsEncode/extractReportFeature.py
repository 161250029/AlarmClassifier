import pandas as pd

from metricsEncode.OneHotEncode import OneHotEncode


class ExtractReportFeature:
    def __init__(self):
        self.one_hot_columns = ['type',  'priority' , 'desc']
        self.normal_columns = ['start']
        self.dataPath = '../prepareData/service/data/program.pkl'
        self.encode = OneHotEncode()

    def extract(self):
        dataframe = pd.read_pickle(self.dataPath)
        res = []
        vectors = []
        for one_hot_column in self.one_hot_columns:
            features = dataframe[one_hot_column].tolist()
            features = [[i] for i in features]
            vector = self.encode.transform(features)
            vectors.append(vector)
            print(len(vector[0]))

        for normal_column in self.normal_columns:
            features = dataframe[normal_column].tolist()
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
    ExtractReportFeature().extract()