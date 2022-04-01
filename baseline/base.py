import abc
import math
import os

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score , matthews_corrcoef


class Base:

    def __init__(self):
        self.root = '/root/GY/Experiment/'
        self.projects = []

        self.project_versions_dict = {}

        self.baseName = None

        self.total_data = None

        # baseline4
        self.train_data = None

        self.train_data_path = None

        self.test_data_path = None

        self.TP = 0

        self.FP = 0

        self.TN = 0

        self.FN = 0

        self.prob_all = []

        self.predict_all = []

        self.label_all = []

        self.EPOCHS = 15


    @abc.abstractmethod
    def train_vocab(self):

        pass

    @abc.abstractmethod
    def train(self , train_data , train_label , test_data , test_label):

        pass

    @abc.abstractmethod
    def vectorlize(self , batch_data):
        """
        向量化

        """
        pass


    @staticmethod
    def visualization(precision, recall, f1, acc , auc , mcc , g_measure , pic_store_path):
        fig = plt.figure()
        x = ['precision', 'recall', 'f1', 'acc' , 'auc' , 'mcc' , 'g_measure']
        y = [precision, recall, f1, acc , auc , mcc , g_measure]
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(x, y)
        plt.title('Numbers of Four eventtypes')
        plt.xlabel('Eventtype')
        plt.ylabel('Number')
        plt.savefig(pic_store_path)

    def calculate_metrics(self):
        precision = self.TP / (self.TP + self.FP)
        recall = self.TP / (self.TP + self.FN)
        f1 = 2 * recall * precision / (recall + precision)
        acc = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        auc = roc_auc_score(self.label_all,self.prob_all)
        mcc = matthews_corrcoef(self.label_all , self.predict_all)
        g_measure = math.sqrt(precision * recall)
        print("TP:{} , TN:{} , FP:{} , FN:{}".format(self.TP, self.TN, self.FP, self.FN))
        print("Precision:{} , Recall:{} , F1:{} , ACC:{} , AUC:{} , MCC:{} , G_Measure:{}".format(precision, recall, f1, acc , auc , mcc , g_measure))
        return self.TP.item(), self.TN.item(), self.FP.item(), self.FN.item(), \
               precision.item(), recall.item(), f1.item(), acc.item() ,auc ,\
               mcc ,g_measure

    def statistic(self , predicted , target , output):
        self.TP += ((predicted == 1) & (target.data == 1)).cpu().sum()
        # TN    predict 和 label 同时为0
        self.TN += ((predicted == 0) & (target.data == 0)).cpu().sum()
        # FN    predict 0 label 1
        self.FN += ((predicted == 0) & (target.data == 1)).cpu().sum()
        # FP    predict 1 label 0
        self.FP += ((predicted == 1) & (target.data == 0)).cpu().sum()

        self.prob_all.extend(
            output[:,1].cpu().detach().numpy())

        self.label_all.extend(target)
        self.predict_all.extend(predicted)

    def clear(self):
        self.total_data = None
        self.train_data = None
        self.train_data_path = None
        self.test_data_path = None
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

        self.prob_all = []
        self.label_all = []
        self.predict_all = []

    def run(self):
        self.init_project_versions()
        for project , versions in self.project_versions_dict.items():
            # 临时加以限制
            # if project not in ['camel-core' , 'cassandra-all']:
            #     continue
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

                self.train(train_data, train_label, test_data, test_label)

                TP, TN, FP, FN, precision, recall, f1, acc, auc, mcc, g_measure = self.calculate_metrics()
                result.append([project , versions[i] , versions[i + 1 ] ,TP, TN, FP, FN, precision, recall, f1, acc, auc, mcc, g_measure])
                pic_file_name = '_'.join([versions[i] , versions[i + 1]])+'.png'
                Base.visualization(precision, recall, f1, acc, auc , mcc, g_measure , os.path.join(baseline_dir_path , pic_file_name))
            dataframe = pd.DataFrame(result)
            dataframe.columns = ["project" , "version1" , "version2" , "TP" , "TN" ,"FP" , "FN" , "precision" , "recall" , "f1" , "acc" , "auc" , "mcc" , "g_measure"]
            dataframe.to_csv(os.path.join(baseline_dir_path , 'result.csv') , index=False)


    def re_run(self):
        self.init_project_versions()
        self.EPOCHS = 30
        for project , versions in self.project_versions_dict.items():
            # 临时加以限制
            if project not in ['camel-core', 'cassandra-all']:
                continue
            result = []
            baseline_dir_path = os.path.join(os.path.join(self.root, project) , self.baseName)
            res_file_path = os.path.join(baseline_dir_path , 'result.csv')
            # 去除索引列
            dataFrame = pd.read_csv(res_file_path , encoding='utf-8')
            print(dataFrame)
            for _, item in dataFrame.iterrows():
                if item.isnull().any():
                    version1 = item['version1']
                    version2 = item['version2']
                    print('project:{} , version:{}-{} start'.format(project, version1, version2))
                    self.clear()
                    train_dir_path = os.path.join(os.path.join(self.root, project), version1)
                    train_file_path = os.path.join(train_dir_path, self.train_data_file_name)
                    self.train_data_path = train_file_path
                    train_data, train_label = self.getData(train_file_path)

                    test_dir_path = os.path.join(os.path.join(self.root, project), version2)
                    test_file_path = os.path.join(test_dir_path, self.test_data_file_name)
                    self.test_data_path = test_file_path
                    test_data, test_label = self.getData(test_file_path)

                    self.total_data = train_data.copy()
                    self.total_data.extend(test_data)
                    self.train_data = train_data
                    print('total:{} , train:{} , test:{}'.format(len(self.total_data), len(train_data), len(test_data)))

                    self.train_vocab()

                    self.train(train_data, train_label, test_data, test_label)

                    TP, TN, FP, FN, precision, recall, f1, acc, auc, mcc, g_measure = self.calculate_metrics()
                    result.append([project, version1, version2, TP, TN, FP, FN, precision, recall, f1, acc, auc, mcc,
                                   g_measure])
                    pic_file_name = '_'.join([version1, version2])+ '.png'
                    Base.visualization(precision, recall, f1, acc, auc, mcc, g_measure,
                                       os.path.join(baseline_dir_path, pic_file_name))
                else:
                    row = item.values.tolist()
                    if len(row) == 15:
                        row.pop(0)
                    result.append(row)
            dataframe = pd.DataFrame(result)
            dataframe.columns = ["project", "version1", "version2", "TP", "TN", "FP", "FN", "precision", "recall",
                                     "f1", "acc", "auc", "mcc", "g_measure"]
            # 不写索引列
            dataframe.to_csv(res_file_path , index=False)

    def split_data(self , tokens, labels):
        data_num = len(tokens)
        train_split = int(8 / 10 * data_num)
        train_data = tokens[:train_split]
        train_label = labels[:train_split]
        test_data = tokens[train_split:]
        test_label = labels[train_split:]
        return train_data, train_label, test_data, test_label


    def init_project_versions(self):
        if (os.path.exists(self.root)):
            files = os.listdir(self.root)
            for file in files:
                m = os.path.join(self.root, file)
                if (os.path.isdir(m) and 'Base' not in file):
                    self.projects.append(file)

        for project in self.projects:
            project_dir = os.path.join(self.root , project)
            if (os.path.exists(project_dir)):
                files = os.listdir(project_dir)
                for file in files:
                    m = os.path.join(project_dir, file)
                    if (os.path.isdir(m) and 'Base' not in file):
                        if self.project_versions_dict.__contains__(project):
                            self.project_versions_dict[project].append(file)
                        else:
                            self.project_versions_dict[project] = [file]

            self.project_versions_dict[project].sort()


