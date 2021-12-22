import os
class Config:
    # 方法体所在目录
    funcSourceDirPath = "D:\\SliceFuncData\\"
    # 新增方法体所在目录
    funcAddSourceDirPath = "D:\\SliceAddFuncData"
    # 特征列
    attribute = ['package' , 'fileName' , 'type' , 'desc' , 'priority' , 'start' , 'end' , 'label' , 'code' ,
                 'lineNum' , 'statementNum' , 'branchStatementNum' , 'callNum' , 'cycleComplexity' , 'depth']
    # 初始源码特征存放路径(源码仅经过切片，ast可编译处理)
    programSourceInfoFilePath = './data/program.pkl'
    # 新增
    programAddSourceInfoFilePath = './data/programAdd.pkl'
    # AST特征存放路径
    programASTFilePath = './data/ast.pkl'
    # 新增
    programAddASTFilePath = './data/astAdd.pkl'
    # 度量属性
    metricsFilePath = os.path.join(funcSourceDirPath , 'metrics.xls')
    # 数据划分比例: train/val/test
    ratio = '2:1:1'
    # 词汇表长度
    vocabLength = 128