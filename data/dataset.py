import os, re
from PIL import Image
from torch.utils import data
import random

from ..utils import conv_to_ndarray
from ..utils import read_from_csv_timeexcluded
from ..utils import read_from_csv_timeexcluded_v3
from ..utils import read_from_csv_v3
from ..utils import read_from_csv_onlymac
from ..utils import tag_to_idx
from ..utils import filter_unused

import torch
import numpy as np
from torchvision import transforms as T
import pandas as pd

from ..config import DefaultConfig

class Dispatcher():

    def __init__(self, csvpaths, usedsize, capacity, windowsize=5, step=5, test=False, filter=None, timeinterval=False):
        if filter is None:
            filter = ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','tag']

        if timeinterval:
            self.traffics = [filter_unused(conv_to_ndarray(read_from_csv_v3(csvpath)), filter, timeinterval=1) for csvpath in csvpaths]
        else:
            self.traffics = [
                filter_unused(conv_to_ndarray(read_from_csv_timeexcluded_v3(csvpath)), filter, timeinterval=0) for csvpath in csvpaths]
        self.cat_num = {}
        for traffic in self.traffics:
            self.cat_num[traffic[0,-1]] = len(traffic)
        self.windowsize = windowsize
        self.windows, self.labels = [], []
        count = dict.fromkeys(range(30), 0)
        if not test:
            """
            train : test = 9 : 1
            """
            #self.traffics = [traffic[:int(0.8*usedsize)] for traffic in self.traffics]
            self.traffics = [traffic[:min(int(0.9*usedsize),int(0.9*len(traffic)))] for traffic in self.traffics]
            #self.windows = [traffic[i:i + self.windowsize, :-1] for traffic in self.traffics for i in
            #                range(len(traffic) / self.windowsize)]
            #self.windows = [traffic[i*step:i*step + self.windowsize, :-1] for traffic in self.traffics for i in range(len(traffic)/step) if i*step+self.windowsize < len(traffic)]
            #self.labels = [traffic[0, -1] for traffic in self.traffics for i in range(len(traffic)/step) if i*step+self.windowsize < len(traffic)]
            for traffic in self.traffics:
                label_set = set(traffic[:,-1])
                for label in label_set:
                    traffic_label = traffic[traffic[:,-1] == label]
                    start = 0
                    while start < len(traffic_label) and start+windowsize < len(traffic_label) and count[label] < capacity:
                        step = random.randint(10,60)
                        self.windows.append(traffic_label[start: start+windowsize, :-1])
                        self.labels.append(traffic_label[0,-1])
                        start += step
                        count[label] += 1

                    """for i in range(len(traffic_label)/step):
                        if i*step+self.windowsize<len(traffic_label) and count[label] < capacity:
                            self.windows.append(traffic_label[i*step:i*step+windowsize,:-1])
                            self.labels.append(traffic_label[0,-1])
                    """

        else:
            #self.traffics = [traffic[int(0.8*usedsize):usedsize] for traffic in self.traffics]
            self.traffics = [traffic[min(int(0.9*usedsize),int(0.9*len(traffic))):] for traffic in self.traffics]
            #self.windows = [traffic[i:i + self.windowsize, :-1] for traffic in self.traffics for i in
            #                range(len(traffic) / self.windowsize)]
            #self.windows = [traffic[i*step:i*step+self.windowsize, :-1] for traffic in self.traffics for i in range(len(traffic)/step) if i*step+self.windowsize < len(traffic)]
            #self.labels = [traffic[0, -1] for traffic in self.traffics for i in range(len(traffic)/step) if i*step+self.windowsize < len(traffic)]            self.windows, self.labels = [], []
            for traffic in self.traffics:
                label_set = set(traffic[:,-1])
                for label in label_set:
                    traffic_label = traffic[traffic[:,-1] == label]
                    start = 0
                    while start < len(traffic_label) and start+windowsize < len(traffic_label) and count[label] < capacity:
                        step = random.randint(10, 60)
                        self.windows.append(traffic_label[start: start+windowsize, :-1])
                        self.labels.append(traffic_label[0,-1])
                        start += step
                        count[label] += 1

        print 'count', count

        self.windows_ = conv_to_ndarray(self.windows)
        self.labels_ = conv_to_ndarray(self.labels)
        order = np.random.permutation(len(self.windows_))
        self.windows_ = self.windows_[order]
        self.labels_ = self.labels_[order]

    def __getitem__(self, index):
        #return {'window':torch.FloatTensor(conv_to_ndarray(self.windows[index])), 'label':self.labels[index]}
        return {'window':torch.FloatTensor(self.windows_[index]), 'label':self.labels_[index]}

    def __len__(self):
        return len(self.windows)

    def stat(self):
        return self.cat_num


class FlowExtractor():
    def __init__(self, csvpaths, usedsize, capacity, windowsize=5, step=5, test=False, filter=None, timeinterval=False):
        if filter is None:
            filter = ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','tag']
        if not timeinterval:
            self.traffics = [filter_unused(conv_to_ndarray(read_from_csv_timeexcluded_v3(csvpath)), filter, timeinterval=0) for csvpath in csvpaths]
        else:
            self.traffics = [filter_unused(conv_to_ndarray(read_from_csv_v3(csvpath)), filter, timeinterval=1) for csvpath in csvpaths]
        self.cat_num = {}
        for traffic in self.traffics:
            self.cat_num[traffic[0,-1]] = len(traffic)
        self.windowsize = windowsize

        count = dict.fromkeys(range(30), 0)

        if not timeinterval:
            if not test: # 9:1
                self.traffics = [traffic[: min(int(0.9*usedsize),int(0.9*len(traffic)))] for traffic in self.traffics]
                """self.windows = [traffic[i * step:i * step + self.windowsize, :-1] for traffic in self.traffics for i in
                                range(len(traffic) / step) if i * step + self.windowsize < len(traffic)]"""
                self.windows, self.labels = [],[]
                for traffic in self.traffics:
                    tag = traffic[0,-1]
                    ip_set = set(traffic[:,0])
                    for ip in ip_set:
                        ip_traffic = traffic[traffic[:,0] == ip]
                        for i in range(len(ip_traffic)/step):
                            if i*step+self.windowsize<len(ip_traffic) and count[tag]<=capacity:
                                self.windows.append(ip_traffic[i*step: i*step+self.windowsize, :-1])
                                self.labels.append(ip_traffic[0, -1])

                                count[tag] += 1

            else:
                self.traffics = [traffic[min(int(0.9*usedsize),int(0.9*len(traffic))) :] for traffic in self.traffics]

                self.windows, self.labels = [],[]
                for traffic in self.traffics:
                    tag = traffic[0, -1]
                    ip_set = set(traffic[:,0])
                    for ip in ip_set:
                        ip_traffic = traffic[traffic[:,0] == ip]
                        for i in range(len(ip_traffic)/step):
                            if i*step+self.windowsize<len(ip_traffic) and count[tag]<=capacity:
                                self.windows.append(ip_traffic[i*step: i*step+self.windowsize, :-1])
                                self.labels.append(ip_traffic[0, -1])

                                count[tag] += 1

        else:
            if not test:  # 9:1
                self.traffics = [traffic[: min(int(0.9 * usedsize), int(0.9 * len(traffic)))] for traffic in
                                 self.traffics]

                self.windows, self.labels = [], []
                for traffic in self.traffics:
                    tag = traffic[0,-1]
                    ip_set = set(traffic[:, 0])
                    for ip in ip_set:
                        ip_traffic = traffic[traffic[:, 0] == ip]
                        for i in range(len(ip_traffic) / step):
                            if i * step + self.windowsize < len(ip_traffic) and count[tag]<=capacity:
                                tmp = ip_traffic[i*step: i*step+self.windowsize, :-1]
                                # if the period of a window exceeds a threshold (5), we do not regard it as a flow.
                                if tmp[-1,-1]-tmp[0,-1]>5:
                                    continue

                                # calculate the difference of timeinterval
                                start_time = tmp[0, -1]
                                for j in range(len(tmp)):
                                    prev = tmp[j, -1]
                                    tmp[j,-1] -= start_time
                                    start_time = prev

                                self.windows.append(tmp)
                                self.labels.append(ip_traffic[0, -1])

                                count[tag] += 1

            else:
                self.traffics = [traffic[min(int(0.9 * usedsize), int(0.9 * len(traffic))) :] for traffic in
                                 self.traffics]
                self.windows, self.labels = [], []

                for traffic in self.traffics:
                    tag = traffic[0,-1]
                    ip_set = set(traffic[:, 0])
                    for ip in ip_set:
                        ip_traffic = traffic[traffic[:, 0] == ip]
                        for i in range(len(ip_traffic) / step):
                            if i * step + self.windowsize < len(ip_traffic) and count[tag]<=capacity:
                                tmp = ip_traffic[i * step: i * step + self.windowsize, :-1]

                                if tmp[-1,-1]-tmp[0,-1]>5:
                                    continue

                                start_time = tmp[0, -1]
                                for j in range(len(tmp)):
                                    prev = tmp[j, -1]
                                    tmp[j, -1] -= start_time
                                    start_time = prev

                                self.windows.append(tmp)
                                self.labels.append(ip_traffic[0, -1])

                                count[tag] += 1

        self.windows_ = conv_to_ndarray(self.windows)
        self.labels_ = conv_to_ndarray(self.labels)

        print 'count = ', count
    def __getitem__(self, index):
        #return {'window':torch.FloatTensor(conv_to_ndarray(self.windows[index])), 'label':self.labels[index]}
        return {'window':torch.FloatTensor(self.windows_[index]), 'label':self.labels_[index]}

    def __len__(self):
        return len(self.windows_)

    def stat(self):
        return self.cat_num


class trafficReader():
    def __init__(self, deviceId, timeinterval=True, rootPath='/home/public/dsk/category_v4_noLAN'):
        opt = DefaultConfig()
        rootPath = os.path.join(rootPath,str(deviceId))
        csvPaths = [os.path.join(rootPath, f) for f in os.listdir(rootPath)]

        if timeinterval:
            self.traffics = [filter_unused(conv_to_ndarray(read_from_csv_v3(csvPath)), opt.mask, timeinterval=1) for
                             csvPath in csvPaths]
        else:
            self.traffics = [
                filter_unused(conv_to_ndarray(read_from_csv_timeexcluded_v3(csvPath)), opt.mask, timeinterval=0) for
                csvPath in csvPaths]

        self.windows, self.labels = [], []
        self.deviceId = deviceId
    def genWindows(self, capacity, windowsize, step):
        for traffic in self.traffics:
            start = 0
            while start+windowsize<len(traffic):
                self.windows.append(traffic[start: start+windowsize, :-1])
                self.labels.append(self.deviceId)
                if len(self.labels)>capacity:
                    break
                start = start+step
        self.numWindows = len(self.windows)
    def removeWindowsLabels(self):
        self.windows, self.labels = [], []
    def getNumofWindows(self):
        return self.numWindows
    def getWindowsLabels(self):
        return zip(self.windows, self.labels)

class trafficReader_onlymac():
    def __init__(self, deviceId, rootPath='/home/public/dsk/category_v4_onlymac'):
        rootPath = os.path.join(rootPath,str(deviceId))
        csvPaths = [os.path.join(rootPath, f) for f in os.listdir(rootPath)]
        self.traffics = [
            conv_to_ndarray(read_from_csv_onlymac(csvPath)) for csvPath in csvPaths
        ]
        self.windows, self.labels = [], []
        self.deviceId = deviceId
    def genWindows(self, capacity, windowsize, step):
        for traffic in self.traffics:
            start = 0
            while start+windowsize<len(traffic):
                self.windows.append(traffic[start: start+windowsize, :-1])
                self.labels.append(self.deviceId)
                if len(self.labels)>capacity:
                    break
                start = start+step
        self.numWindows = len(self.windows)
    def removeWindowsLabels(self):
        self.windows, self.labels = [], []
    def getNumofWindows(self):
        return self.numWindows
    def getWindowsLabels(self):
        return zip(self.windows, self.labels)

class trafficContainer():
    def __init__(self, windows, labels):
        self.windows, self.labels = windows, labels
    def __getitem__(self, index):
        return {'window':torch.FloatTensor(self.windows[index]), 'label':self.labels[index]}
    def __len__(self):
        return len(self.windows)

if __name__ == "__main__":
    windowsizes = [50, 100, 200, 300, 400]
    df = pd.DataFrame(columns=[str(dev) for dev in range(30)], index=[str(x) for x in windowsizes])
    for dev in range(30):
        for windowsize in windowsizes:
            step = windowsize/2
            tr = trafficReader(dev,True)
            tr.genWindows(windowsize, step)
            df[str(dev)][str(windowsize)] = tr.getNumofWindows()
    print df




