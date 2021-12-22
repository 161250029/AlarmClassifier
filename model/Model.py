import abc

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score


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
