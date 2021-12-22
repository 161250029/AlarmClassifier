from baseline import utils

class StrConvertService:
    def __init__(self , s):
        self.s = s
    def convert(self):
        s_snake = self.s
        try:
            s_snake = utils.name_convert_to_snake(self.s)
        except:
            pass
        s_snake_split = s_snake.split('_')
        return s_snake_split

if __name__ == '__main__':
    print(StrConvertService('UserLoginCount').convert())
    print(StrConvertService('user_login_Count').convert())