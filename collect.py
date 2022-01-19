import os
import pandas as pd



class Collect:
    def __init__(self):
        self.root = '/root/GY/Experiment/'
        self.projects = []
        self.bases = ['BaseLine1', 'BaseLine2', 'BaseLine3'
            , 'BaseLine4', 'BaseLine5']
        self.result_file_name = 'result.csv'

    def run(self):
        self.init_project()


        total_data_frame = pd.DataFrame(columns=['project', 'version1', 'version2',
                                                 'TP' , 'TN' , 'FP' , 'FN' , 'precision' ,
                                                 'recall' , 'f1' , 'acc' , '方法'])

        total_info_frame = pd.DataFrame(columns=['project', 'version', '正报' , '误报' , '总数量'])
        for project in self.projects:
            project_dir_path = os.path.join(self.root , project)
            for base in self.bases:
                base_dir_path = os.path.join(project_dir_path , base)

                result_file_path = os.path.join(base_dir_path , self.result_file_name)

                dataFrame = pd.read_csv(result_file_path)
                list = []
                for i in range(len(dataFrame)):
                    list.append(base)
                dataFrame['方法'] = list
                total_data_frame = pd.concat([dataFrame,total_data_frame],ignore_index=True)
        total_data_frame.to_csv('/root/GY/experiment.csv')


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
        total_info_frame.to_csv('/root/GY/info.csv')


    def init_project(self):
        if (os.path.exists(self.root)):
            files = os.listdir(self.root)
            for file in files:
                m = os.path.join(self.root, file)
                if (os.path.isdir(m) and 'Base' not in file):
                    self.projects.append(file)


Collect().run()
