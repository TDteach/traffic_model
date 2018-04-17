from ..config import DefaultConfig
from dataset import Dispatcher
from dataset import FlowExtractor
from dataset_pd import FlowExtractor_time
from dataset_pd import genStatisVec
import cPickle as pickle
import os
import threading

def genfixedWindow_nonFlow(opt, windowsize, capacity, test=False):
    dataset = Dispatcher(opt.csv_paths, opt.used_size, capacity, windowsize, opt.step, test, opt.mask, True)
    if test:
        filePath = os.path.join('/home/4tshare/iot/infocomm_data/generated/','nonFlow_'+str(windowsize)+'_'+str(capacity)+'_test'+'.pkl')
    else:
        filePath = os.path.join('/home/4tshare/iot/infocomm_data/generated/','nonFlow_'+str(windowsize)+'_'+str(capacity)+'_nontest'+'.pkl')
    print 'filePath', filePath
    tmp = (dataset.windows_, dataset.labels_)
    pickle.dump(tmp, open(filePath, 'w'))
    print 'pickle dumping ends'

def genfixedWindow_Flow(opt, windowsize, capacity, test=False):
    dataset = FlowExtractor_time(opt.csv_paths, opt.used_size, capacity, windowsize, opt.step, test, opt.mask)
    if test:
        filePath = os.path.join('/home/4tshare/iot/infocomm_data/generated/','Flow_'+str(windowsize)+'_'+str(capacity)+'_test'+'.pkl')
    else:
        filePath = os.path.join('/home/4tshare/iot/infocomm_data/generated/','Flow_'+str(windowsize)+'_'+str(capacity)+'_nontest'+'.pkl')
    print 'filePath', filePath
    tmp = (dataset.windows_, dataset.labels_)
    pickle.dump(tmp, open(filePath, 'w'))
    print 'pickle dumping ends'

class myThread (threading.Thread):
    def __init__(self, opt, windowsize, capacity, test):
        super(myThread, self).__init__(self)
        self.windowsize = windowsize
        self.capacity = capacity
        self.test = test
        self.opt = opt
    def run(self):
        genfixedWindow_nonFlow(self.opt, self.windowsize, self.capacity, self.test)
        print self.windowsize, self.capacity, self.test, 'finished....'

if __name__ == "__main__":
    opt = DefaultConfig()
    threads = [threading.Thread(target=genfixedWindow_nonFlow, args=(opt, i, 20000, True)) for i in range(10, 210, 10)]
    threads.extend([threading.Thread(target=genfixedWindow_nonFlow, args=(opt, i, 20000, True)) for i in range(10, 210, 10)])

    for i in range(len(threads)):
        print 'thread ', i, 'start'
        threads[i].start()

    for i in range(len(threads)):
        threads[i].join()
