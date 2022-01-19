import os
import pandas as pd

class preProcess:
    def __init__(self , dirPath , metricsPath):
        self.dirPath = dirPath
        self.storePath = 'token.pkl'
        self.metricsPath = metricsPath
        self.attribute = ['package' , 'fileName' , 'type' , 'desc' , 'priority' , 'start' , 'end' , 'label' , 'token' ,
                 'lineNum' , 'statementNum' , 'branchStatementNum' , 'callNum' , 'cycleComplexity' , 'depth']
    def toPickle(self):
        fileNames = self.readDir()
        metrics = self.readMetrics()
        features = [fileName.split('#') for fileName in  fileNames]
        lables = [feature[len(feature) - 1].split('.')[0] for feature in features]
        fileTexts = [open(os.path.join(self.dirPath ,fileName)).read() for fileName in fileNames]
        for i in range(0 , len(features)):
            feature = features[i]
            fileName = fileNames[i]
            feature[len(feature) - 1] = lables[i]
            feature.append(fileTexts[i][:-1].split(','))
            metricsFeature = metrics[metrics['fileName'] == fileName]
            feature.append(metricsFeature.iloc[0]['lineNum'])
            feature.append(metricsFeature.iloc[0]['statementNum'])
            feature.append(metricsFeature.iloc[0]['branchStatementNum'])
            feature.append(metricsFeature.iloc[0]['callNum'])
            feature.append(metricsFeature.iloc[0]['cycleComplexity'])
            feature.append(metricsFeature.iloc[0]['depth'])
        dataFrame = pd.DataFrame(features)
        dataFrame.columns = self.attribute
        dataFrame.to_pickle(self.storePath)

    def readSource(self):
        dataFrame = pd.read_pickle(self.storePath)
        return dataFrame

    def readMetrics(self):
        dataFrame = pd.read_excel(self.metricsPath)
        return dataFrame


    def readDir(self):
        res = []
        for root,dirs,files in os.walk(self.dirPath):
            for file in  files:
                if file.endswith('java'):
                    res.append(file)
        return res

    def run(self):
        self.toPickle()

if __name__ == '__main__':
    # preProcess().run()
    preProcess = preProcess('D:\ASTFeature' , 'D:\SliceFuncData\\metrics.xls')
    preProcess.run()
    print(preProcess.readSource())


