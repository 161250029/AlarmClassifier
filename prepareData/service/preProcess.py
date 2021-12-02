import os
import pandas as pd
from prepareData.config import Config
import javalang

class preProcess:
    def toPickle(self):
        fileNames = self.readDir()
        features = [fileName.split('#') for fileName in  fileNames]
        lables = [feature[len(feature) - 1].split('.')[0] for feature in features]
        fileTexts = [open(os.path.join(Config.funcSourceDirPath ,fileName)).read() for fileName in fileNames]
        for i in range(0 , len(features)):
            feature = features[i]
            feature[len(feature) - 1] = lables[i]
            feature.append(fileTexts[i])
        dataFrame = pd.DataFrame(features)
        dataFrame.columns = Config.attribute
        dataFrame.to_pickle(Config.programSourceInfoFilePath)

    def parse(self , func):
        tokens = javalang.tokenizer.tokenize(func)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        return tree

    def generateFeatures(self):
        dataFrame = pd.read_pickle(Config.programSourceInfoFilePath)
        dataFrame['code'] = dataFrame['code'].apply(self.parse)
        dataFrame.to_pickle(Config.programASTFilePath)

    def readASTFeatures(self):
        dataFrame = pd.read_pickle(Config.programASTFilePath)
        return dataFrame

    def readSourceCode(self):
        dataFrame = pd.read_pickle(Config.programSourceInfoFilePath)
        return dataFrame


    def readDir(self):
        res = []
        for root,dirs,files in os.walk(Config.funcSourceDirPath):
            for file in  files:
                if file.endswith('java'):
                    res.append(file)
        return res

    def run(self):
        self.toPickle()
        self.generateFeatures()

if __name__ == '__main__':
    preProcess().run()
    print(preProcess().readASTFeatures())


