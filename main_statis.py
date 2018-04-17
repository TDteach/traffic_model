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
import sklearn
from statistic import genStatisVec
import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import svm

"""
['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','tag']
"""
"""
In this module, we add timeinterval into the model.
"""
window_lens = []

def train(**kwargs):
    """

    :param kwargs:
    :return: model --> the SVM model trained by training set.
     transformer --> a transformer using StandardScaler.
    """
    opt = DefaultConfig()
    opt.parse(kwargs)

    #train_data = FlowExtractor_time(opt.csv_paths, opt.used_size, opt.capacity, opt.window_size, opt.step, filter=opt.mask)
    #val_data = FlowExtractor_time(opt.csv_paths, opt.used_size, opt.capacity , opt.window_size, opt.step, test=True, filter=opt.mask)
    train_data = Dispatcher(opt.csv_paths, opt.used_size, opt.capacity, opt.window_size, opt.step,
                            filter=['ip', 'port', 'ipv4', 'ipv6', 'tcp', 'udp', 'http', 'ssl', 'dns', 'direction1',
                                    'direction2', 'datalen', 'timeinterval', 'tag'], timeinterval=True)
    val_data = Dispatcher(opt.csv_paths, opt.used_size, opt.capacity, opt.window_size, opt.step, test=True,
                          filter=['ip', 'port', 'ipv4', 'ipv6', 'tcp', 'udp', 'http', 'ssl', 'dns', 'direction1',
                                  'direction2', 'datalen', 'timeinterval', 'tag'], timeinterval=True)
    train_dataloader = DataLoader(train_data,batch_size=1,shuffle=True,drop_last=True)#,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,batch_size=1,shuffle=False,drop_last=True)

    print 'len(train_data) = ', len(train_data)

    scaler = StandardScaler()
    clf = svm.SVC()
    vecs, labels = [], []
    for i, data in enumerate(train_dataloader):
        input, label = data['window'].numpy(), data['label'].numpy()
        #print 'input.shape', input.shape
        #print 'input[0].shape', input[0].shape
        ipt = input[0]
        for j in range(len(ipt)):
            start_time = ipt[j][0, -1]
            for k in range(len(ipt[j])):
                prev = ipt[j][k, -1]
                ipt[j][k, -1] -= start_time
                start_time = prev

        vec = genStatisVec(ipt, True, True)

        #print 'vec', vec
        vecs.append(vec)
        labels.append(label)
    vecs = conv_to_ndarray(vecs)
    labels = conv_to_ndarray(labels).squeeze()

    scaler.fit(vecs)
    new_vecs = scaler.transform(vecs)

    clf.fit(new_vecs, labels.ravel())

    validate(clf, scaler, val_dataloader)

    return clf, scaler

def validate(clf, scaler, dataloader):
    opt = DefaultConfig()
    total, correct = 0, 0
    error_d = dict.fromkeys(range(30), 0)
    total_d = dict.fromkeys(range(30), 0)

    confusion_matrix = CM.ConfusionMatrix(30)
    confusion_matrix.reset()

    new_inputs, labels = [], []
    for i, data in enumerate(dataloader):
        input, label = data['window'].numpy(), data['label'].numpy()
        new_inputs.append(genStatisVec(input[0], True, True))
        labels.append(label)

        total += 1

    labels = conv_to_ndarray(labels).squeeze()
    new_inputs = scaler.transform(conv_to_ndarray(new_inputs))
    results = conv_to_ndarray(clf.predict(new_inputs))
    print 'results', results
    print 'labels', labels

    for j in range(len(results)):
        confusion_matrix.add(results[j], labels[j])
        if results[j] == labels[j]:
            correct += 1
    for e in labels:
        total_d[e] += 1
    error = labels[labels!=results] #val_label.data[val_label.data != predicted]
    for e in error:
        error_d[e] += 1

    """
    Some operations on my confusion matrix...
    """

    acc_rate = 1.0 * correct / total
    print 'accuracy = ', acc_rate
    for k in range(30):
        print 'k = ', k, error_d[k], total_d[k]
    keys = range(30)
    dData = {'parameter':str(opt.max_epoch)+'|'+str(opt.lr)+'|'+str(opt.window_size)+'|'+','.join(opt.mask),
             'algorithm':'SVM',
             'validating total':{k:total_d[k] for k in keys},
             'validating error':{k:error_d[k] for k in keys},
             'accuracy':acc_rate}
    save_path = '/home/4tshare/iot/dev/traffic_model/results/' + time.strftime('%m%d_%H:%M:%S')
    save_result(save_path, dData)

    print 'CM Dropcam', confusion_matrix[3]
    print 'CM Android Phone', confusion_matrix[7]
    print 'CM Macbook', confusion_matrix[17]
    print 'CM Laptop', confusion_matrix[16]
    print 'CM Galaxy Tab', confusion_matrix[2]
    print 'CM Amazon Echo', confusion_matrix[13]
    print 'CM Iphone', confusion_matrix[23]
    """
    for images, labels in test_loader:
    images = Variable(images.view(-1, sequence_length, input_size))
    outputs = rnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    """
    return confusion_matrix, acc_rate

if __name__ == "__main__":
    print 'torch.__version__', torch.__version__
    train()


