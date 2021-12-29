import os
import pandas as pd

from baseline.Config import Config


class preProcess:
    def __init__(self , mode):
        self.sliceTokenDirPath = Config.sliceTokenDirPath
        self.sliceTokenStorePath = Config.sliceTokenStorePath

        self.byteTokenDirPath = Config.byteTokenDirPath
        self.byteTokenStorePath = Config.byteTokenStorePath

        self.methodTokenDirPath = Config.methodTokenDirPath
        self.methodTokenStorePath = Config.methodTokenStorePath

        self.metricsPath = Config.metricsPath

        self.workDirPath = None
        self.workStorePath = None


        self.attribute = Config.attribute

        self.mode = mode

    def toPickle(self):
        if self.mode == 'slice':
            self.workDirPath = self.sliceTokenDirPath
            self.workStorePath = self.sliceTokenStorePath
        elif self.mode == 'byte':
            self.workDirPath = self.byteTokenDirPath
            self.workStorePath = self.byteTokenStorePath
        elif self.mode == 'method':
            self.workDirPath = self.methodTokenDirPath
            self.workStorePath = self.methodTokenStorePath
        fileNames = self.readDir()
        features = [fileName.split('#') for fileName in  fileNames]
        lables = [feature[len(feature) - 1].split('.')[0] for feature in features]
        fileTexts = [open(os.path.join(self.workDirPath ,fileName)).read() for fileName in fileNames]
        for i in range(0 , len(features)):
            feature = features[i]
            feature[len(feature) - 1] = lables[i]
            feature.append(fileTexts[i])
        dataFrame = pd.DataFrame(features)
        dataFrame.columns = self.attribute
        dataFrame.to_pickle(self.workStorePath)

    def readSource(self):
        dataFrame = pd.read_pickle(self.workStorePath)
        return dataFrame

    def readMetrics(self):
        dataFrame = pd.read_excel(self.metricsPath)
        return dataFrame


    def readDir(self):
        res = []
        for root,dirs,files in os.walk(self.workDirPath):
            for file in  files:
                if file.endswith('java'):
                    res.append(file)
        return res

    def run(self):
        self.toPickle()

if __name__ == '__main__':
    # preProcess = preProcess('slice')
    # preProcess.run()
    # print(preProcess.readSource())
    preProcess = preProcess('byte')
    preProcess.run()
    print(preProcess.readSource())

    # preProcess = preProcess('method')
    # preProcess.run()
    # print(preProcess.readSource())


