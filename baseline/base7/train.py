import os
import torch
import pandas as pd
from baseline.base import Base

from baseline.base7.extractMetricsFeature import ExtractMetricsFeature
from baseline.base7.linear import Linear


class BaseLine7(Base):
    def __init__(self):
        super(BaseLine7, self).__init__()

        self.metrics_csv_dir = '/root/GY/ThesisNewData/feature/'
        self.report_csv_dir = '/root/GY/ThesisNewData/report/'
        self.projects = ['camel-core' , 'cassandra-all' , 'hadoop-common'
                         , 'lucene-core' , 'solr-core']
        self.baseName = 'BaseLine7'

    def run(self):
        for project in self.projects:
            metrics_csv_path = os.path.join(self.metrics_csv_dir , project + '.csv')
            report_csv_path = os.path.join(self.report_csv_dir , project + '.csv')
            metrics_dict = ExtractMetricsFeature(metrics_csv_path , report_csv_path).extract_feature()
            versions = sorted(metrics_dict.keys())
            result = []

            if project not in ['camel-core', 'cassandra-all']:
                continue
            else:
                self.check_label(project, metrics_dict)
            baseline_dir_path = os.path.join(os.path.join(self.root, project), self.baseName)
            if not os.path.exists(baseline_dir_path):
                os.makedirs(baseline_dir_path)
            for i in range(len(versions) - 1):
                # 清除上次实验数据
                self.clear()
                train_version = versions[i]
                train_df = metrics_dict.get(train_version)
                train_data = list(train_df['metrics'])
                train_label = torch.LongTensor(list(train_df['label']))

                test_version = versions[i + 1]
                test_df = metrics_dict.get(test_version)
                test_data = list(test_df['metrics'])
                test_label = torch.LongTensor(list(test_df['label']))
                print('train:{} , test:{}'.format(len(train_data) , len(test_data)))
                self.train(train_data , train_label , test_data , test_label)

                TP, TN, FP, FN, precision, recall, f1, acc, auc, mcc, g_measure = self.calculate_metrics()
                result.append(
                    [project, train_version, test_version, TP, TN, FP, FN, precision, recall, f1, acc, auc, mcc,
                     g_measure])
            dataframe = pd.DataFrame(result)
            dataframe.columns = ["project", "version1", "version2", "TP", "TN", "FP", "FN", "precision", "recall", "f1",
                                 "acc", "auc", "mcc", "g_measure"]
            dataframe.to_csv(os.path.join(baseline_dir_path, 'result.csv'))

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

    def train(self, train_data, train_label, test_data, test_label):
        train_features = train_data
        test_features = test_data


        hidden_dim = len(train_features[0])
        output_size = 2

        model = Linear(hidden_dim, output_size)
        parameters = model.parameters()
        optimizer = torch.optim.Adadelta(parameters)
        loss_function = torch.nn.CrossEntropyLoss()

        EPOCHS = 15
        batch_size = 64
        for epcho in range(EPOCHS):
            i = 0
            while i < len(train_features):
                cur_train_features = train_features[i:i + batch_size]
                cur_train_labels = train_label[i:i + batch_size]

                optimizer.zero_grad()

                output = model(cur_train_features)
                loss = loss_function(output, cur_train_labels)
                print('epcho:{} , loss:{}'.format(epcho, loss))
                loss.backward()
                optimizer.step()
                i += batch_size
        i = 0
        while i < len(test_features):
            cur_test_features = test_features[i: i + batch_size]
            cur_test_labels = test_label[i: i + batch_size]
            output = model(cur_test_features)
            cur_test_labels = torch.LongTensor(cur_test_labels)
            _, predicted = torch.max(output, 1)
            self.statistic(predicted, cur_test_labels, output)
            i += batch_size


if __name__ == '__main__':
    baseline7 = BaseLine7()
    baseline7.run()



