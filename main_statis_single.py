import models
from config import DefaultConfig
from data import Dispatcher
from data import FlowExtractor
from data import FlowExtractor_time
from utils import tag_to_idx
from utils import conv_to_ndarray
from utils import save_result
from torch.utils.data import DataLoader
import torch
import torch.autograd as autograd
import os
import numpy as np
from utils import confusion_matrix as CM
from utils import read_from_csv_v3
import time
import random
from statistic import genStatisVec
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from data import gen_mixture

"""
['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','tag']
"""
"""
In this module, we add timeinterval into the model.
"""
window_lens = []

svm_vec = [svm.SVC() for i in range(30)]

def train_single(svm, training_set, label, capacity):
    """

    :param svm: the svm model
    :param training_set: zip format: [([a,b,c],0), ([d,e,f],1), ...]
    :return:
    """
    pos_set = [list(elem) for elem in training_set if elem[-1] == label] #training_set[[training_set[:,-1] == label]]
    neg_set = [list(elem) for elem in training_set if elem[-1] != label] #training_set[[training_set[:,-1] != label]]

    if len(pos_set) == 0:
        return None
    #python2 and python3 handle zip differently, in python3 zip <zip at 0x7fxxxxx>

    pos_set = np.array(random.sample(pos_set, min(capacity, len(pos_set))))
    neg_set = np.array(random.sample(neg_set, min(capacity, len(neg_set))))

    for elem in pos_set:
        elem[-1] = 1
    for elem in neg_set:
        elem[-1] = 0
    comb = np.concatenate((pos_set, neg_set), axis=0)
    np.random.shuffle(comb)
    svm.fit(comb[:,0].tolist(), comb[:,-1].tolist())
    # Remember to use tolist()
    return svm


def train(svm_vec):
    """

    :param kwargs:
    :return: model --> the SVM model trained by training set.
     transformer --> a transformer using StandardScaler.
    """
    opt = DefaultConfig()

    train_data = FlowExtractor_time(opt.csv_paths, opt.used_size, opt.capacity, opt.window_size, opt.step, filter=opt.mask)
    val_data = FlowExtractor_time(opt.csv_paths, opt.used_size, opt.capacity , opt.window_size, opt.step, test=True, filter=opt.mask)
    train_dataloader = DataLoader(train_data,batch_size=1,shuffle=True,drop_last=True)#,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,batch_size=1,shuffle=False,drop_last=True)

    print 'len(train_data) = ', len(train_data)

    stat = dict.fromkeys(range(30), 0)

    scaler = StandardScaler()

    vecs, labels = [], []
    """for i, data in enumerate(train_dataloader):
        input, label = data['window'].numpy(), data['label'].numpy()
        #print 'input.shape', input.shape
        #print 'input[0].shape', input[0].shape
        vec = genStatisVec(input[0])

        window_lens.append(len(input[0]))

        #print 'vec', vec
        vecs.append(vec)
        labels.append(label)
    """
    print 'len(train_data.windows_)', len(train_data.windows_)
    print 'len(train_data.labels_)', len(train_data.labels_)

    vecs = train_data.windows
    #labels = [label[0] for label in labels]

    for i in range(len(vecs)):
        start_time = vecs[i][0, -1]
        for j in range(len(vecs[i])):
            prev = vecs[i][j,-1]
            vecs[i][j,-1] -= start_time
            start_time = prev

    labels = train_data.labels
    new_vecs = [genStatisVec(vec) for vec in vecs]
    scaler.fit(new_vecs)
    new_vecs = scaler.transform(new_vecs)

    training_set = zip(new_vecs, labels)
    for i in range(30):
        svm_vec[i] = train_single(svm_vec[i], training_set, i, opt.capacity)

    validate(val_data)
    #train_single(svm_vec[3], training_set, 3, 3000)

def validate(flowextractor):
    opt = DefaultConfig()
    total, correct = 0, 0
    error_d = dict.fromkeys(range(30), 0)
    total_d = dict.fromkeys(range(30), 0)

    scaler = StandardScaler()

    windows, labels = flowextractor.windows_, flowextractor.labels_
    print 'validate--> len(windows)', len(windows)
    print 'validate--> len(labels)', len(labels)
    validate_set = gen_mixture(windows, labels, opt.capacity, 4, algorithm="svm")
    fs,ls = zip(*validate_set)
    scaler.fit(fs)
    fs = scaler.transform(fs)

    correct, false_p, false_n = 0, 0, 0
    for j in range(len(fs)):
        tmp_res = set([k for k in range(len(svm_vec)) if svm_vec[k] and svm_vec[k].predict(np.array(fs[j]).reshape(1,-1))[0] == 1])
        m = tmp_res & set(ls[j])
        p = tmp_res - set(ls[j])
        n = set(ls[j]) - tmp_res

        correct += len(m)
        false_p += len(p)
        false_n += len(n)

        total += len(ls[j])

    print "false_positive", 1.0*false_p/total
    print "false_negative", 1.0*false_n/total
    print "precise", 1.0*correct/total





if __name__ == "__main__":
    print 'torch.__version__', torch.__version__
    train(svm_vec)


