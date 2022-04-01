import os

import pandas as pd

from Tool.FileTool import FileTool


class CheckLabel:

    def __init__(self):
        self.root = '/root/GY'
        self.projects = ['camel-core', 'cassandra-all']
        self.old_root = '/root/GY/Experiment/'


    def check(self):
        for project in self.projects:
            new_data_file = os.path.join(self.root , '.'.join([project , 'csv']))
            new_data_frame = self.read_csv(new_data_file)
            old_project_dir_path = os.path.join(self.old_root , project)
            version_list = self.get_versions(old_project_dir_path)
            for version in version_list:
                old_file_list = FileTool().get_file_by_ext(os.path.join(old_project_dir_path , version) , ['pkl'])
                for old_file in old_file_list:
                    old_data_frame = self.read_pkl(old_file)
                    for _, item in old_data_frame.iterrows():
                        label_list = new_data_frame.loc[(new_data_frame["start"] == int(item["start"])) & (
                                    new_data_frame["version"] == item["version"]) & (new_data_frame["fileName"] == item[
                            "fileName"])].isPositive.tolist()
                        if len(label_list) != 0:
                            item['label'] = str(label_list[0]).lower()
                    old_data_frame.to_pickle(old_file)
                    print(self.read_pkl(old_file))

    def get_versions(self , project_dir):
        versions = []
        if (os.path.exists(project_dir)):
            files = os.listdir(project_dir)
            for file in files:
                m = os.path.join(project_dir, file)
                if (os.path.isdir(m) and 'Base' not in file):
                    versions.append(file)
        return versions

    def read_csv(self , csv_path):
        return pd.read_csv(csv_path)

    def read_pkl(self , pkl_path):
        return pd.read_pickle(pkl_path)

if __name__ == '__main__':
    check = CheckLabel()
    check.check()