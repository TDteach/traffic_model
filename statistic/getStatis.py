import pandas as pd
import sklearn
import numpy as np
from .. import config
from ..utils import read_params

def genStatisVec(array, ip=False, port=False):
    #df = pd.DataFrame(array, columns=['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','timeinterval'])
    def add(a,b):
        return a+b
    opt = config.DefaultConfig()
    df = pd.DataFrame(array, columns=opt.mask[:-1])
    """
    feature_vec: [min, max, mean, median absolute deviation, standard deviation, variance, skew, kurtosis, percentiles]
    """
    ip_dict, port_dict = read_params()
    if ip and port:
        one_hot_ip = np.array([0] * (len(ip_dict)/10+1))
        one_hot_port = np.array([0] * (len(port_dict)/10+1))
        for elem in set(df['ip']):
            one_hot_ip[int(elem/10)] += 1
        for elem in set(df['port']):
            one_hot_port[int(elem/10)] += 1
        one_hot = np.concatenate((one_hot_ip, one_hot_port))

    elif ip:
        one_hot = np.array([0]*(len(ip_dict)/10+1))
        for elem in set(df['ip']):
            one_hot[int(elem/10)] += 1
    elif port:
        one_hot = np.array([0]*(len(port_dict)/10+1))
        for elem in set(df['port']):
            one_hot[int(elem/10)] += 1
    else:
        one_hot = np.array([])

    df_d0, df_d1 = df[df['direction1'] == 1], df[df['direction2'] == 1]
    fea_vec = np.array([
        df_d0['datalen'].min(), df_d0['datalen'].max(), df_d0['datalen'].mean(), df_d0['datalen'].std(), df_d0['datalen'].skew(), df_d0['datalen'].kurt(),
        df_d1['datalen'].min(), df_d1['datalen'].max(), df_d1['datalen'].mean(), df_d1['datalen'].std(), df_d1['datalen'].skew(), df_d1['datalen'].kurt(),
        df['timeinterval'].min(), df['timeinterval'].max(), df['timeinterval'].mean(), df['timeinterval'].std(), df['timeinterval'].skew(), df['timeinterval'].kurt(),
        #df['direction1'].sum(), df['direction2'].sum()
    ])

    res = np.concatenate((one_hot, fea_vec))
    return np.nan_to_num(res)


def genStatisVec2(array, ip=False, port=False):
    opt = config.DefaultConfig()
    df = pd.DataFrame(array, columns=opt.mask[:-1])
    ip_dict, port_dict = read_params()
    if ip and port:
        if ip and port:
            one_hot_ip = np.array([0] * (len(ip_dict) / 10 + 1))
            one_hot_port = np.array([0] * (len(port_dict) / 10 + 1))
            for elem in set(df['ip']):
                one_hot_ip[int(elem / 10)] += 1
            for elem in set(df['port']):
                one_hot_port[int(elem / 10)] += 1
            one_hot = np.concatenate((one_hot_ip, one_hot_port))

        elif ip:
            one_hot = np.array([0] * (len(ip_dict) / 10 + 1))
            for elem in set(df['ip']):
                one_hot[int(elem / 10)] += 1
        elif port:
            one_hot = np.array([0] * (len(port_dict) / 10 + 1))
            for elem in set(df['port']):
                one_hot[int(elem / 10)] += 1
        else:
            one_hot = np.array([])
    elif ip:
        one_hot = np.array([0]*(len(ip_dict)/10+1))
        for elem in set(df['ip']):
            one_hot[int(elem/10)] += 1
    elif port:
        one_hot = np.array([0]*(len(port_dict)/10+1))
        for elem in set(df['port']):
            one_hot[int(elem/10)] += 1
    else:
        one_hot = np.array([])

    df['datalen'] = df['datalen']*df['direction1']-df['datalen']*df['direction2']
    fea_vec = np.array([
        df['datalen'].min(), df['datalen'].max(), df['datalen'].mean(), df['datalen'].std(), df['datalen'].skew(), df['datalen'].kurt(),
        df['timeinterval'].min(), df['timeinterval'].max(), df['timeinterval'].mean(), df['timeinterval'].std(),
        df['timeinterval'].skew(), df['timeinterval'].kurt(),
    ])
    res = np.concatenate((one_hot, fea_vec))
    return np.nan_to_num(res)

def genStatisVec3(array, ip=False, port=False):
    """

    :param array:
    :param ip:
    :param port:
    :return:
    """
    opt = config.DefaultConfig()
    df = pd.DataFrame(array, columns=opt.mask[:-1])
    ip_dict, port_dict = read_params()
    if ip and port:
        if ip and port:
            one_hot_ip = np.array([0] * (len(ip_dict)+ 1))
            one_hot_port = np.array([0] * (len(port_dict)+ 1))
            for elem in set(df['ip']):
                one_hot_ip[int(elem)] += 1
            for elem in set(df['port']):
                one_hot_port[int(elem)] += 1
            one_hot = np.concatenate((one_hot_ip, one_hot_port))

        elif ip:
            one_hot = np.array([0] * (len(ip_dict)+ 1))
            for elem in set(df['ip']):
                one_hot[int(elem)] += 1
        elif port:
            one_hot = np.array([0] * (len(port_dict)+ 1))
            for elem in set(df['port']):
                one_hot[int(elem)] += 1
        else:
            one_hot = np.array([])
    elif ip:
        one_hot = np.array([0]*(len(ip_dict)+1))
        for elem in set(df['ip']):
            one_hot[int(elem)] += 1
    elif port:
        one_hot = np.array([0]*(len(port_dict)+1))
        for elem in set(df['port']):
            one_hot[int(elem)] += 1
    else:
        one_hot = np.array([])

    df['datalen'] = df['datalen']*df['direction1']-df['datalen']*df['direction2']
    fea_vec = np.array([
        df['datalen'].min(), df['datalen'].max(), df['datalen'].mean(), df['datalen'].std(), df['datalen'].skew(), df['datalen'].kurt(),
        df['timeinterval'].min(), df['timeinterval'].max(), df['timeinterval'].mean(), df['timeinterval'].std(),
        df['timeinterval'].skew(), df['timeinterval'].kurt(),
    ])
    res = np.concatenate((one_hot, fea_vec))
    return np.nan_to_num(res)