import numpy as np

class ConfusionMatrix(object):

    def __init__(self, num_class):
        self.conf = np.ndarray((num_class, num_class), dtype=np.int32)
        self.n_class = num_class

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, label):
        assert predicted < self.n_class
        assert label < self.n_class
        self.conf[int(label), int(predicted)] += 1

    def __getitem__(self, item):
        return self.conf[item, :]
