import pandas as pd
class ExtractMetricsFeature:
    def __init__(self , metrics_path , report_path):
        self.feature_data_frame = pd.read_csv(metrics_path , index_col='index')
        self.report_data_frame = pd.read_csv(report_path , index_col='index')
        self.clear_data_frame()
        self.delete_str_metrics()
        self.merge()
        # print(self.feature_data_frame)
        # print(self.report_data_frame)
        self.data_frame = pd.concat([self.feature_data_frame,self.report_data_frame],axis=1,join='inner')
        # print(self.data_frame)
        self.metrics_dict = {}

    def split_data_frame(self):
        for version , df in self.data_frame.groupby('version'):
            self.metrics_dict.update({version:df})

    def get_version_data_frame(self):
        self.extract_feature()
        return self.metrics_dict

    def extract_feature(self):
        self.split_data_frame()
        temp = self.metrics_dict.copy()
        for version in temp.keys():
            labels = temp[version]['label'].values.tolist()
            count = 0
            # 去除全误报的版本数据
            for i in range(len(labels)):
                if labels[i] == 0:
                    count += 1
            if count == len(labels):
                self.metrics_dict.pop(version)
        return self.metrics_dict

    def clear_data_frame(self):
        self.report_data_frame.drop('next', axis=1, inplace=True)
        self.report_data_frame.drop('label', axis=1, inplace=True)

    def delete_str_metrics(self):
        for column in self.feature_data_frame.columns:
            if self.feature_data_frame[column].dtype == object:
                self.feature_data_frame.drop(column, axis=1, inplace=True)

    def merge(self):
        columns = list(self.feature_data_frame.columns)
        columns.pop(len(columns) - 1)
        self.feature_data_frame['metrics'] = self.feature_data_frame[columns].values.tolist()



if __name__ == '__main__':
    handle = ExtractMetricsFeature('/root/GY/ThesisNewData/feature/hadoop-common.csv' , '/root/GY/ThesisNewData/report/hadoop-common.csv')
    handle.extract_feature()