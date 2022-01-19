import os

import javalang
import pandas as pd

from baseline.Config import Config


class preProcess:
    def __init__(self):

        self.root = '/root/GY/Data/project/'

        self.store_dir = '/root/GY/Experiment/'

        self.sliceTokenName = 'sliceToken.pkl'

        self.byteTokenName = 'byteToken.pkl'

        self.fiveLinesTokenName = 'fiveLines.pkl'

        self.funcBodyName = 'func.pkl'

        self.astName = 'ast.pkl'

        self.projects = []

        self.metricsPath = Config.metricsPath

        self.attribute = Config.attribute

    def init_projects(self):
        if (os.path.exists(self.root)):
            # 获取该目录下的所有文件或文件夹目录
            files = os.listdir(self.root)
            for file in files:
                # 得到该文件下所有目录的路径
                m = os.path.join(self.root , file)
                # 判断该路径下是否是文件夹
                if (os.path.isdir(m)):
                    self.projects.append(file)

    def version_dict(self , dir_path):
        fileNames = self.readDir(dir_path)
        features = [fileName.split('#') for fileName in fileNames]
        dict = {}
        for i in range(len(features)):
            feature = features[i]
            version = feature[1]
            if dict.__contains__(version):
                dict[version].append(fileNames[i])
            else:
                dict[version] = [fileNames[i]]
        return dict

    def toPickle(self , dir_path , store_file_name):
        dict = self.version_dict(dir_path)
        for version, fileNames in dict.items():
            features = [fileName.split('#') for fileName in  fileNames]
            lables = [feature[len(feature) - 1].split('.')[0] for feature in features]
            fileTexts = [open(os.path.join(dir_path, fileName)).read() for fileName in fileNames]
            for i in range(len(features)):
                feature = features[i]
                feature[len(feature) - 1] = lables[i]
                feature.append(fileTexts[i])
            dataFrame = pd.DataFrame(features)
            dataFrame.columns = self.attribute
            store_project_dir_path = os.path.join(self.store_dir , features[0][0])
            store_project_version_dir_path = os.path.join(store_project_dir_path , version)
            if not os.path.exists(store_project_version_dir_path):
                os.makedirs(store_project_version_dir_path)
            store_path = os.path.join(store_project_version_dir_path , store_file_name)
            dataFrame.to_pickle(store_path)


    def parse(self , func):
        tokens = javalang.tokenizer.tokenize(func)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        return tree

    def generateFeatures(self , func_file_path):
        dataFrame = pd.read_pickle(func_file_path)
        asts = []
        index = []
        for _, item in dataFrame.iterrows():
            try:
                tree = self.parse(item["code"])
                asts.append(tree)
            except Exception as ex:
                print("出现如下异常%s"%ex)
                index.append(_)
                continue
        # dataFrame['code'] = dataFrame['code'].apply(self.parse)
        dataFrame.drop(index=index , inplace=True)

        # 索引重排很重要
        dataFrame = dataFrame.reset_index(drop=True)
        dataFrame['code'] = pd.Series(asts)
        return dataFrame

    def generate_ast(self):
        projects = []
        if (os.path.exists(self.store_dir)):
            files = os.listdir(self.store_dir)
            for file in files:
                m = os.path.join(self.store_dir , file)
                if (os.path.isdir(m)):
                    projects.append(file)

        for project in projects:
            versions = []
            project_dir_path = os.path.join(self.store_dir , project)
            if (os.path.exists(project_dir_path)):
                files = os.listdir(project_dir_path)
                for file in files:
                    m = os.path.join(project_dir_path, file)
                    if (os.path.isdir(m)):
                        versions.append(file)
            for version in versions:
                version_dir_path = os.path.join(project_dir_path , version)
                func_file_path = os.path.join(version_dir_path , self.funcBodyName)
                dataFrame = self.generateFeatures(func_file_path)
                print(dataFrame)
                ast_file_path = os.path.join(version_dir_path , self.astName)
                dataFrame.to_pickle(ast_file_path)



    def readSource(self , store_path):
        dataFrame = pd.read_pickle(store_path)
        return dataFrame

    def readMetrics(self):
        dataFrame = pd.read_excel(self.metricsPath)
        return dataFrame


    def readDir(self , dir_path):
        res = []
        for root,dirs,files in os.walk(dir_path):
            for file in files:
                if file.endswith('java'):
                    res.append(file)
        return res

    def run(self):
        self.init_projects()
        for project in self.projects:
            project_dir_path = os.path.join(self.root , project)
            project_store_dir_path = os.path.join(self.store_dir, project)
            if not os.path.exists(project_store_dir_path):
                os.mkdir(project_store_dir_path)

            project_byte_token_dir_path = os.path.join(project_dir_path , 'ByteToken')
            self.toPickle(project_byte_token_dir_path , self.byteTokenName)

            project_slice_token_dir_path = os.path.join(project_dir_path , 'SliceToken')
            self.toPickle(project_slice_token_dir_path , self.sliceTokenName)

            project_five_lines_dir_path = os.path.join(project_dir_path , 'FiveLines')
            self.toPickle(project_five_lines_dir_path , self.fiveLinesTokenName)

            project_func_body_dir_path = os.path.join(project_dir_path , 'ReplaceSliceFunc')
            self.toPickle(project_func_body_dir_path , self.funcBodyName)

if __name__ == '__main__':
    Process1 = preProcess()
    Process1.generate_ast()
    # print(Process3.readSource())


