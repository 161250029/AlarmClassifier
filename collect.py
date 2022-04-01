import math
import os
import pandas as pd


# 汇总所有的excel结果
class Collect:
    def __init__(self):
        self.root = '/root/GY/Experiment/'
        self.projects = []
        self.bases = ['BaseLine1', 'BaseLine2', 'BaseLine3'
            , 'BaseLine4', 'BaseLine5' , 'BaseLine6']
        self.desc = {'BaseLine1': '字节码+Word+LSTM' ,
                     'BaseLine2': '上下五行+Word+CNN',
                     'BaseLine3': '程序切片+Word+LSTM',
                     'BaseLine4': 'ASTNN+DL',
                     'BaseLine5': 'ASTNN+度量+DL',
                     'BaseLine6': 'ASTNN+Word+DL',
                    }

        self.model_feature_bases = ['BaseLine4', 'BaseLine5' , 'BaseLine7' , 'BaseLine8'
                                    ,'BaseLine9' , 'BaseLine10']
        self.model_feature_desc = {
                     'BaseLine4': 'ASTNN+DL',
                     'BaseLine5': 'ASTNN+度量+DL',
                     'BaeLine7': '度量+DL',
                     'BaseLine8': 'ASTNN+传统',
                     'BaseLine9': 'ASTNN+度量+传统',
                     'BaseLine10': '度量+传统',
                    }
        self.result_file_name = 'result.csv'

        self.model_feature_store_path = '/root/GY/model_feature.csv'
        self.baseline_store_path = '/root/GY/experiment.csv'

    def run(self):
        self.init_project()
        total_data_frame = pd.DataFrame(columns=['project', 'version1', 'version2',
                                                 'TP' , 'TN' , 'FP' , 'FN' , 'precision' ,
                                                 'recall' , 'f1' , 'acc' , 'auc' , 'mcc', 'g_measure', '方法'])

        total_model_feature_data_frame = pd.DataFrame(columns=['project', 'version1', 'version2',
                                                 'TP', 'TN', 'FP', 'FN', 'precision',
                                                 'recall', 'f1', 'acc', 'auc', 'mcc', 'g_measure', 'model','方法'])

        total_info_frame = pd.DataFrame(columns=['project', 'version', '正报' , '误报' , '总数量'])
        for project in self.projects:
            project_dir_path = os.path.join(self.root , project)
            for base in self.bases:
                base_dir_path = os.path.join(project_dir_path , base)

                result_file_path = os.path.join(base_dir_path , self.result_file_name)

                dataFrame = pd.read_csv(result_file_path)
                base_name = []
                base_desc = []
                for i in range(len(dataFrame)):
                    base_name.append(base)
                    base_desc.append(self.desc.get(base))
                dataFrame['方法'] = base_name
                dataFrame['描述'] = base_desc
                total_data_frame = pd.concat([dataFrame,total_data_frame],ignore_index=True)
            total_data_frame.to_csv(self.baseline_store_path, index=False)

            for model_feature_base in self.model_feature_bases:
                base_dir_path = os.path.join(project_dir_path, model_feature_base)

                result_file_path = os.path.join(base_dir_path, self.result_file_name)

                dataFrame = pd.read_csv(result_file_path)
                base_name = []
                base_desc = []
                for i in range(len(dataFrame)):
                    base_name.append(model_feature_base)
                    base_desc.append(self.model_feature_desc.get(model_feature_base))
                dataFrame['方法'] = base_name
                dataFrame['描述'] = base_desc
                total_model_feature_data_frame = pd.concat([dataFrame, total_model_feature_data_frame], ignore_index=True)
            total_model_feature_data_frame.to_csv('/root/GY/model_feature.csv', index=False)


        for project in self.projects:
            project_dir_path = os.path.join(self.root , project)
            baseline1_dir_path = os.path.join(project_dir_path, 'BaseLine1')
            result_file_path = os.path.join(baseline1_dir_path, self.result_file_name)
            data = pd.read_csv(result_file_path)
            temp = pd.DataFrame(columns=['project', 'version', '正报', '误报', '总数量'])
            temp['project'] = data['project']
            temp['version'] = data['version2']
            temp['正报'] = data['TP'] + data['FN']
            temp['误报'] = data['TN'] + data['FP']
            temp['总数量'] = temp['正报'] + temp['误报']
            total_info_frame = pd.concat([total_info_frame, temp], ignore_index=True)
        total_info_frame.to_csv('/root/GY/info.csv' , index=False)

    def get_best_result(self):
        dataframe = pd.read_csv(self.baseline_store_path)
        dataframe.drop(axis=1, columns=['Unnamed: 0'], inplace=True)
        res = []
        for _ , baselines_data in dataframe.groupby(['project']):
            for baseline , baseline_data in baselines_data.groupby(['方法']):
                cur_row = baseline_data.iloc[0]
                for index , item in baseline_data.iterrows():
                    if math.isnan(cur_row['f1']):
                        cur_row = item
                        continue
                    temp_f1 = item['f1']
                    if temp_f1 > cur_row['f1']:
                        cur_row = item
                temp = list(cur_row)
                res.append(temp)
        best_result = pd.DataFrame(res , columns=['project', 'version1', 'version2',
                                                 'TP' , 'TN' , 'FP' , 'FN' , 'precision' ,
                                                 'recall' , 'f1' , 'acc' , 'auc' , 'mcc', 'g_measure', '方法' , '描述'])
        best_result.to_csv('/root/GY/bestbaseline.csv' , index=False)

    def get_best_model_result(self):
        dataframe = pd.read_csv('/root/GY/model_feature.csv')
        dataframe.drop(axis=1, columns=['Unnamed: 0'], inplace=True)
        res = []
        for _, baselines_data in dataframe.groupby(['project']):
            for baseline, baseline_data in baselines_data.groupby(['方法' , 'model']):
                cur_row = baseline_data.iloc[0]
                for index, item in baseline_data.iterrows():
                    if math.isnan(cur_row['f1']):
                        cur_row = item
                        continue
                    temp_f1 = item['f1']
                    if temp_f1 > cur_row['f1']:
                        cur_row = item
                temp = list(cur_row)
                res.append(temp)
        best_result = pd.DataFrame(res, columns=['project', 'version1', 'version2',
                                                 'TP', 'TN', 'FP', 'FN', 'precision',
                                                 'recall', 'f1', 'acc', 'auc', 'mcc', 'g_measure', 'model','方法' , '描述'])
        best_result.to_csv('/root/GY/bestmodel.csv', index=False)


    def init_project(self):
        if (os.path.exists(self.root)):
            files = os.listdir(self.root)
            for file in files:
                m = os.path.join(self.root, file)
                if (os.path.isdir(m) and 'Base' not in file):
                    self.projects.append(file)
collect = Collect()
# collect.run()
# collect.get_best_result()
collect.get_best_model_result()