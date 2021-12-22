from sklearn.preprocessing import OneHotEncoder
import sklearn

class OneHotEncode:
    # features: 二维数组
    def __init__(self , features):
        self.features = features
        self.onehot = OneHotEncoder(handle_unknown='ignore')

    def transform(self):
        print(self.features)
        return self.onehot.fit_transform(self.features).toarray()

if __name__ == '__main__':
    model = OneHotEncode([['Male' , 1], ['NONE' , 3], ['Female' , 5] , ['test' , 7] ,['Male' , 1]])
    print(model.transform())

    enc = OneHotEncoder()
    enc.fit([[0, 0, 3],
             [1, 1, 0],
             [0, 2, 1],
             [1, 0, 2]])


    ans = enc.transform([[0, 1, 3]]).toarray()
    print(ans)
