import os


class ModelConfig:
    root = '/root/GY/AlarmClassifier/CodeFeatureModel/'
    vocab_model_path = os.path.join(root, 'w2vModel.model')
    model_paramenter_path = os.path.join(root , 'code.pt')