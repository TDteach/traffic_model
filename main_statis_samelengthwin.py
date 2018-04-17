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

model = svm.SVC(decision_function_shape='ovr')

def train_single(svm, training_set):
    samples, labels = zip(*training_set)
    svm.fit(samples, labels)
    # Remember to use tolist()
    return svm


def train(model):
    """

    :param kwargs:
    :return: model --> the SVM model trained by training set.
     transformer --> a transformer using StandardScaler.
    """
    opt = DefaultConfig()
    train_data = Dispatcher(opt.csv_paths, opt.used_size, opt.capacity, opt.window_size, opt.step,
                            filter=['ip', 'port', 'ipv4', 'ipv6', 'tcp', 'udp', 'http', 'ssl', 'dns', 'direction1',
                                    'direction2', 'datalen', 'timeinterval', 'tag'], timeinterval=True)
    val_data = Dispatcher(opt.csv_paths, opt.used_size, opt.capacity, opt.window_size, opt.step, test=True,
                          filter=['ip', 'port', 'ipv4', 'ipv6', 'tcp', 'udp', 'http', 'ssl', 'dns', 'direction1',
                                  'direction2', 'datalen', 'timeinterval', 'tag'], timeinterval=True)

    print 'len(train_data) = ', len(train_data)
    scaler = StandardScaler()

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
    new_vecs = [genStatisVec(vec, True, True) for vec in vecs]
    scaler.fit(new_vecs)
    new_vecs = scaler.transform(new_vecs)

    training_set = zip(new_vecs, labels)

    model = train_single(model, training_set)

    validate(model, train_data)
    validate(model, val_data, 2)
    validate(model, val_data, 4)
    #train_single(svm_vec[3], training_set, 3, 3000)

def validate(model, extractor, mix=1):
    opt = DefaultConfig()
    total, correct = 0, 0

    scaler = StandardScaler()

    windows, labels = extractor.windows_, extractor.labels_
    print 'validate--> len(windows)', len(windows)
    print 'validate--> len(labels)', len(labels)
    if mix == 1:
        validate_set = random.sample(zip(windows, labels), opt.capacity)
    else:
        validate_set = gen_mixture(windows, labels, opt.capacity, mix, algorithm="svm", ip=True, port=True)

    if mix == 1:
        fs, ls = zip(*validate_set)
        fs = list(fs)
        ls = list(ls)
        for i in range(len(fs)):
            start_time = fs[i][0, -1]
            for j in range(len(fs[i])):
                prev = fs[i][j, -1]
                fs[i][j, -1] -= start_time
                start_time = prev

        for i in range(len(ls)):
            if type(ls[i]) == np.float64:
                ls[i] = [ls[i]]

        new_vecs = [genStatisVec(f, True, True) for f in fs]
        scaler.fit(new_vecs)
        fs = scaler.transform(new_vecs)
    else:
        fs,ls = zip(*validate_set)
        fs = list(fs)
        ls = list(ls)
        for i in range(len(ls)):
            if type(ls[i]) == np.float64:
                ls[i] = [ls[i]]
        scaler.fit(fs)
        fs = scaler.transform(fs)

    correct, false_p, false_n = 0, 0, 0
    for j in range(len(fs)):
        #tmp_res = set([k for k in range(len(svm_vec)) if svm_vec[k] and svm_vec[k].predict(np.array(fs[j]).reshape(1,-1))[0] == 1])
        tmp_res = zip(model.decision_function([fs[j]])[0].tolist(), range(30))
        """
        The type of svm.decision_function([x]) is numpy.ndarray-->array([[x x x x]]), therefore if you zip(decision_function([x]), range(len(x)))
        you will get [(array([x x x x]),0)]
        """
        tmp_res.sort(key=lambda x:x[0], reverse=True)
        tmp_res = conv_to_ndarray(tmp_res[:mix])[:,1]
        m = set(tmp_res) & set(ls[j])
        p = set(tmp_res) - set(ls[j])
        n = set(ls[j]) - set(tmp_res)

        correct += len(m)
        false_p += len(p)
        false_n += len(n)

        total += len(ls[j])

    print "false_positive", 1.0*false_p/total
    print "false_negative", 1.0*false_n/total
    print "precise", 1.0*correct/total





if __name__ == "__main__":
    print 'torch.__version__', torch.__version__
    train(model)


