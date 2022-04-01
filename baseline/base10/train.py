import os
import torch
import pandas as pd
from baseline.base import Base

from baseline.base10.extractMetricsFeature import ExtractMetricsFeature

from model import DecisionTreeModel, KNNModel, SvmModel, RFModel


class BaseLine10(Base):
    def __init__(self):
        super(BaseLine10, self).__init__()

        self.metrics_csv_dir = '/root/GY/ThesisNewData/feature/'
        self.report_csv_dir = '/root/GY/ThesisNewData/report/'
        self.projects = ['camel-core', 'cassandra-all', 'hadoop-common'
            , 'lucene-core', 'solr-core']
        self.baseName = 'BaseLine10'

        self.model_list = [DecisionTreeModel(), KNNModel(),
                           SvmModel(), RFModel()]



    def run(self):
        for project in self.projects:
            metrics_csv_path = os.path.join(self.metrics_csv_dir, project + '.csv')
            report_csv_path = os.path.join(self.report_csv_dir, project + '.csv')
            metrics_dict = ExtractMetricsFeature(metrics_csv_path, report_csv_path).extract_feature()
            versions = sorted(metrics_dict.keys())
            if project in ['camel-core', 'cassandra-all']:
                self.check_label(project , metrics_dict)
            result = []
            baseline_dir_path = os.path.join(os.path.join(self.root, project), self.baseName)
            if not os.path.exists(baseline_dir_path):
                os.makedirs(baseline_dir_path)
            for i in range(len(versions) - 1):
                self.clear()
                train_version = versions[i]
                train_df = metrics_dict.get(train_version)
                train_data = list(train_df['metrics'])
                train_label = torch.LongTensor(list(train_df['label']))

                test_version = versions[i + 1]
                test_df = metrics_dict.get(test_version)
                test_data = list(test_df['metrics'])
                test_label = torch.LongTensor(list(test_df['label']))
                print('train:{} , test:{}'.format(len(train_data), len(test_data)))
                statistics = self.train(train_data, train_label, test_data, test_label)

                model_name = ["DecisionTreeModel", "KNNModel",
                              "RFModel"]
                for j in range(len(statistics)):
                    statistics[j].insert(0, versions[i + 1])
                    statistics[j].insert(0, versions[i])
                    statistics[j].insert(0, project)
                    statistics[j].append(model_name[j])
                result.extend(statistics)
            dataframe = pd.DataFrame(result)
            dataframe.columns = ["project", "version1", "version2", "TP", "TN", "FP", "FN", "precision", "recall", "f1",
                                 "acc", "auc", "mcc", "g_measure", "model"]
            dataframe.to_csv(os.path.join(baseline_dir_path, 'result.csv'))

    # 针对camel-core、cassenal-all数据需要校对标签
    def check_label(self , project , version_data_frame):
        project_csv_path = os.path.join('/root/GY' , project + '.csv')
        new_data_frame = pd.read_csv(project_csv_path)
        for version in version_data_frame.keys():
            for _, item in version_data_frame[version].iterrows():
                label_list = new_data_frame.loc[(new_data_frame["start"] == int(item["location"])) & (
                        new_data_frame["version"] == version) & (new_data_frame["sourceFile"] == item[
                        "path"])].isPositive.tolist()
                if len(label_list) != 0:
                    if str(label_list[0]).lower() == 'false':
                        # 替换单元值
                        version_data_frame[version].at[_ ,'label'] = 0
                    else:
                        version_data_frame[version].at[_ , 'label'] = 1
            print(version_data_frame[version]['label'])

    def train(self, train_data, train_label, test_data, test_label):
        train_features = train_data
        test_features = test_data
        res = []
        for model in self.model_list:
            model.train(train_features, train_label)
            statistic = list(model.get_result_statistics(test_features, test_label))
            res.append(statistic)
        return res


if __name__ == '__main__':
    baseline10 = BaseLine10()
    baseline10.run()
