import cPickle as pickle
import os
def pickleoneFile(filePath):
    (windows, labels) = pickle.load(open(filePath))
    return (windows, labels)

class Dataloader():
    def __init__(self, windowsize, capacity, test):
        filePath = os.path.join('/home/4tshare/iot/infocomm_data/generated',
                                'nonFlow_' + str(windowsize) + '_' + str(capacity) + '_' + tmp)
        (windows, labels) = pickleoneFile(filePath)
        self.windows, self.labels = windows, labels