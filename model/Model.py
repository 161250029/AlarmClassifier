import abc

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score , matthews_corrcoef



class Model:

    @abc.abstractmethod
    def train(self, x_train, y_train):
        """
        模型训练

        :param x_train: 训练数据
        :param y_train: 训练标签
        """
        pass

    @abc.abstractmethod
    def predict(self, x_test):
        """
        返回预测值

        :param x_test: 预测数据
        :return: 预测值
        """
        pass

    @abc.abstractmethod
    def predict_proba(self, x_test):
        """
        返回预测为正值的概率

        :param x_test: 预测数据
        :return: 预测为正概率
        """
        pass

    def accuracy_score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)

    def recall_score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return recall_score(y_test , y_predict)

    def precision_score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return precision_score(y_test, y_predict)

    def get_result_statistics(self , x_test , y_test):
        predictions = self.predict(x_test)
        proba = self.predict_proba(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        print("TN, FP, FN, TP: ", (tn, fp, fn, tp))
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        acc = (tp + tn) / (tp + tn + fp + fn)
        auc = roc_auc_score(y_test, proba)
        mcc = matthews_corrcoef(y_test , predictions)

        PF = fp / (fp + tn)
        f1 = 2 * recall * precision / (recall + precision)
        g_measure = 2 * recall * (1 - PF) / (recall + 1 - PF)
        # print("recall:{} , precision:{} , acc:{} , f1:{} , mcc:{},auc:{}".format(recall , precision , acc , f1 , mcc , auc))
        return tp, tn, fp, fn, precision, recall, f1, acc , auc , mcc ,g_measure
