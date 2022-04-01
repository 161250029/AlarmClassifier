from CodeFeatureModel.utils import get_blocks_v1 as func

class SplitAST:
    def __init__(self , vocab_model , train_data: list):
        self.vocab_model = vocab_model
        self.train_data = train_data

    def vectorlize(self):
        vocab = self.vocab_model.wv.vocab
        max_token = self.vocab_model.wv.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        features = []
        for data in self.train_data:
            features.append(trans2seq(data))
        return features



