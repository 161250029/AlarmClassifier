import os
import sys
import pandas as pd

from pandas import DataFrame

import javalang


# 将警告方法体转换成AST,保存为pkl形式
class PreProcess:

    def __init__(self, task_dir_path: str):
        self.task_dir_path = task_dir_path
        self.func_body_dir_path = os.path.join(self.task_dir_path , 'FuncBody')
        self.attribute = ['fileName', 'type', 'catogray', 'priority', 'start', 'end', 'code']

    def preprocess(self) -> DataFrame:
        fileNames = self.readDir(self.func_body_dir_path)
        dataframe = self.prepross_report_info(fileNames)
        dataframe = self.generateASTs(dataframe)
        dataframe.to_pickle(os.path.join(self.task_dir_path , 'ast.pkl'))
        return dataframe

    def generateASTs(self, dataframe: DataFrame) -> list:
        asts = []
        index = []
        for _, item in dataframe.iterrows():
            try:
                tree = self.parse(item["code"])
                asts.append(tree)
            except Exception as ex:
                print("出现如下异常%s" % ex)
                index.append(_)
                continue
        dataframe.drop(index=index, inplace=True)
        # 索引重排很重要
        dataframe = dataframe.reset_index(drop=True)
        dataframe['ast'] = pd.Series(asts)
        return dataframe

    def prepross_report_info(self, fileNames: list) -> DataFrame:
        features = [fileName.split('#') for fileName in fileNames]
        fileTexts = [open(os.path.join(self.func_body_dir_path, fileName)).read() for fileName in fileNames]
        for i in range(len(features)):
            feature = features[i]
            feature[len(feature) - 1] = feature[len(feature) - 1].split('.')[0]
            feature.append(fileTexts[i])
        dataFrame = pd.DataFrame(features)
        dataFrame.columns = self.attribute
        return dataFrame

    def parse(self, func: str):
        tokens = javalang.tokenizer.tokenize(func)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        return tree

    def readDir(self, dir_path: str) -> list:
        res = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('java'):
                    res.append(file)
        return res

if __name__ == '__main__':
    a = []
    for i in range(1, len(sys.argv)):
        a.append((sys.argv[i]))
    preprocess = PreProcess(a[0])
    dataframe = preprocess.preprocess()
    print(dataframe.columns)
