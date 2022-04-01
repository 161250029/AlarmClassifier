import os

import pandas as pd

from baseline.base import Base
from model import DecisionTreeModel, KNNModel, SvmModel, RFModel


class BaseLine8(Base):

    def __init__(self):
        super(BaseLine8, self).__init__()

        self.train_data_file_name = 'trainBestFeatures.pkl'
        self.test_data_file_name = 'testBestFeatures.pkl'

        self.vocab_model = None

        self.baseName = 'BaseLine8'

        self.model_list = [DecisionTreeModel() , KNNModel() ,
                           RFModel()]

    def getData(self , data_path):
        data = pd.read_pickle(data_path)
        tokens = []
        labels = []
        for _, item in data.iterrows():
            astnn = item['astnn']
            tokens.append(astnn)
            label = 1 if item['label'] == 'true' else 0
            labels.append(label)
        return tokens, labels

    def vectorlize(self, batch_data):
        pass

    def train_vocab(self):
        pass

    def run(self):
        self.init_project_versions()
        for project , versions in self.project_versions_dict.items():
            result = []
            baseline_dir_path = os.path.join(os.path.join(self.root, project) , self.baseName)
            if not os.path.exists(baseline_dir_path):
                os.makedirs(baseline_dir_path)
            for i in range(len(versions) - 1):
                print('project:{} , version:{}-{} start'.format(project, versions[i] , versions[i + 1]))
                # 清除上次实验数据
                self.clear()
                train_dir_path = os.path.join(os.path.join(self.root , project) , versions[i])
                train_file_path = os.path.join(train_dir_path , self.train_data_file_name)
                self.train_data_path = train_file_path
                train_data, train_label = self.getData(train_file_path)

                test_dir_path = os.path.join(os.path.join(self.root , project) , versions[i + 1])
                test_file_path = os.path.join(test_dir_path , self.test_data_file_name)
                self.test_data_path = test_file_path
                test_data, test_label = self.getData(test_file_path)

                self.total_data = train_data.copy()
                self.total_data.extend(test_data)
                self.train_data = train_data
                print('total:{} , train:{} , test:{}'.format(len(self.total_data) , len(train_data) , len(test_data)))

                self.train_vocab()
                statistics = self.train(train_data, train_label, test_data, test_label)

                model_name = ["DecisionTreeModel", "KNNModel",
                 "RFModel", "SvmModel"]
                for j in range(len(statistics)):
                    statistics[j].insert(0 , versions[i + 1])
                    statistics[j].insert(0, versions[i])
                    statistics[j].insert(0, project)
                    statistics[j].append(model_name[j])
                result.extend(statistics)
            dataframe = pd.DataFrame(result)
            dataframe.columns = ["project" , "version1" , "version2" , "TP" , "TN" ,"FP" , "FN" , "precision" , "recall" , "f1" , "acc" , "auc" , "mcc" , "g_measure" , "model"]
            dataframe.to_csv(os.path.join(baseline_dir_path , 'result.csv'))

    # 针对camel-core、cassenal-all数据需要校对标签
    def check_label(self, project, version_data_frame):
        project_csv_path = os.path.join('/root/GY', project + '.csv')
        new_data_frame = pd.read_csv(project_csv_path)
        for version in version_data_frame.keys():
            for _, item in version_data_frame[version].iterrows():
                label_list = new_data_frame.loc[(new_data_frame["start"] == int(item["location"])) & (
                        new_data_frame["version"] == version) & (new_data_frame["sourceFile"] == item[
                    "path"])].isPositive.tolist()
                if len(label_list) != 0:
                    if str(label_list[0]).lower() == 'false':
                        # 替换单元值
                        version_data_frame[version].at[_, 'label'] = 0
                    else:
                        version_data_frame[version].at[_, 'label'] = 1
            print(version_data_frame[version]['label'])

    def train(self , train_data, train_label, test_data, test_label):
        train_features = train_data
        test_features = test_data

        res = []
        for model in self.model_list:
            model.train(train_features , train_label)
            statistic = list(model.get_result_statistics(test_features , test_label))
            res.append(statistic)
        return res

if __name__ == '__main__':
    baseline8 = BaseLine8()
    baseline8.run()



