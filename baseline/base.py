import abc


class Base:
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

    def split_data(self , tokens, labels):
        data_num = len(tokens)
        train_split = int(8 / 10 * data_num)
        train_data = tokens[:train_split]
        train_label = labels[:train_split]
        test_data = tokens[train_split:]
        test_label = labels[train_split:]
        return train_data, train_label, test_data, test_label