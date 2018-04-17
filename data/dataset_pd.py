import pandas as pd
from ..utils import filter_unused
from ..utils import read_from_csv_timeexcluded_v3
from ..utils import read_from_csv_v3
from ..utils import conv_to_ndarray
from ..utils import merge_flows
from ..utils import merge_flows_hard
from ..statistic import genStatisVec
from ..statistic import genStatisVec2
from ..statistic import genStatisVec3
import torch
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

class FlowExtractor_time():
    def __init__(self, csvpaths, usedsize, capacity, time_thres=2.0, windowsize=5, step=5, test=False, filter=None):
        if filter is None:
            filter = ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','tag']

        self.traffics = [filter_unused(conv_to_ndarray(read_from_csv_v3(csvpath)), filter, timeinterval=1) for
                             csvpath in csvpaths]
        self.cat_num = {}
        for traffic in self.traffics:
            self.cat_num[traffic[0, -1]] = len(traffic)

        self.windowsize = windowsize
        count = dict.fromkeys(range(30), 0)

        if not test:
            self.traffics = [traffic[: min(int(0.9*usedsize),int(0.9*len(traffic)))] for traffic in self.traffics]
            self.windows, self.labels = [], []

            for traffic in self.traffics:
                tag = traffic[0, -1]
                ip_set = set(traffic[:,0])
                for ip in ip_set:
                    ip_traffic = traffic[traffic[:,0] == ip]

                    start, end = 0, 0
                    while(start <= end and end < len(ip_traffic) and count[tag] < capacity):
                        if ip_traffic[end, -2]-ip_traffic[start, -2]>=time_thres:
                            tmp = ip_traffic[start:end, :-1]

                            """start_time = tmp[0, -1]
                            for j in range(len(tmp)):
                                prev = tmp[j, -1]
                                tmp[j, -1] -= start_time
                                start_time = prev
                            """

                            self.windows.append(tmp)
                            self.labels.append(ip_traffic[0, -1])

                            """
                            8/3/2018 Here I modify 'start = end' to 'start += random.randint(1, end-start)
                            """
                            #start = end
                            start += random.randint(1, end-start)
                            end = start+1

                            count[tag] += 1
                            if count[tag]>capacity:
                                break
                        else:
                            end += 1
        else:
            self.traffics = [traffic[min(int(0.9 * usedsize), int(0.9 * len(traffic))) :] for traffic in
                             self.traffics]
            self.windows, self.labels = [], []

            for traffic in self.traffics:
                tag = traffic[0, -1]
                ip_set = set(traffic[:,0])
                for ip in ip_set:
                    ip_traffic = traffic[traffic[:,0] == ip]

                    start, end = 0, 0
                    while(start <= end and end < len(ip_traffic) and count[tag] < capacity):
                        if ip_traffic[end, -2]-ip_traffic[start, -2]>=time_thres:
                            tmp = ip_traffic[start:end, :-1]

                            self.windows.append(tmp)
                            self.labels.append(ip_traffic[0, -1])

                            start = end

                            count[tag] += 1
                            if count[tag]>capacity:
                                break
                        else:
                            end += 1
        self.windows_ = conv_to_ndarray(self.windows)
        self.labels_ = conv_to_ndarray(self.labels)

        #np.random.shuffle(self.windows_)
        #np.random.shuffle(self.labels_)

        order = np.random.permutation(len(self.windows_))
        self.windows_ = self.windows_[order]
        self.labels_ = self.labels_[order]

        print 'count = ', count

    def __getitem__(self, index):
        return {'window':torch.FloatTensor(self.windows_[index]), 'label':self.labels_[index]}

    def __len__(self):
        return len(self.windows_)


def gen_mixture(windows, labels, num_mix, num_involve, algorithm='nn', ip=False, port=False):
    """

    :param windows: all the flow windows
    :param labels: all the labels
    :param num_mix: how many new mixture will be generated
    :param num_involve:
    :return:
    """
    fs = zip(windows, labels)

    res = []

    for i in range(num_mix):
        sample = random.sample(fs, num_involve)
        (f,l) = merge_flows(sample)
        if algorithm == 'svm':
            sf = []
            step = len(f)/num_mix
            start = 0
            while start<len(f):
                sf.append(genStatisVec3(f[start:start+step], ip, port))
                start += step
            #sf = genStatisVec(f, ip, port)
        else:
            sf = f
        res.append((sf,l))

    return res

def gen_hardmix(windows, labels, num_mix, num_involve, algorithm='nn', ip=False, port=False):

    fs = zip(windows, labels)
    res = []
    for i in range(num_mix):
        sample = random.sample(fs, num_involve)
        (f,l) = merge_flows_hard(sample)
        if algorithm == 'svm':
            sf = []
            step = len(f)/num_involve
            start = 0
            while start<len(f):
                sf.append(genStatisVec2(f[start:start+step], ip, port))
                start += step
            #sf = genStatisVec(f, ip, port)
        else:
            sf = f
        res.append((sf, l))
    return res


