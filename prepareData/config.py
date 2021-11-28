class Config:
    # 方法体所在目录
    funcSourceDirPath = "D:\\SliceFuncData\\"
    # 特征列
    attribute = ['package' , 'fileName' , 'type' , 'desc' , 'priority' , 'start' , 'end' , 'label' , 'code']
    # 初始源码特征存放路径(源码仅经过切片，ast可编译处理)
    programSourceInfoFilePath = './data/program.pkl'
    # AST特征存放路径
    programASTFilePath = './data/ast.pkl'
    # 数据划分比例: train/val/test
    ratio = '2:1:1'
    # 词汇表长度
    vocabLength = 128