from sklearn.neural_network import MLPRegressor

from model import Model


class MLPModel(Model):

    def __init__(self):
        '''第一个隐藏层有100个节点，第二层有50个，激活函数用relu，梯度下降方法用adam'''
        self.model = MLPRegressor(hidden_layer_sizes=(100,50), activation='relu',solver='adam',
        alpha=0.01,max_iter=200)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict_proba(self, x_test):
        pos_index = list(self.model.classes_).index(1)
        return self.model.predict_proba(x_test)[:pos_index]

    def predict(self, x_test):
        return self.model.predict(x_test)

if __name__ == '__main__':
    x = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    y = [0, 0, 0, 1, 1, 1, 1 , 1 , 1]
    neigh = MLPModel()
    neigh.train(x , y)
    x_test = [[0.5] , [1.5] , [2.5] , [3.5] , [4.5] , [5.5]]
    print(neigh.predict(x_test))