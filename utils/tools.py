import json,os
import numpy as np
import pandas as pd
import random

def read_from_csv(csv):
    traffic = []
    #org, domain, protocol, port, direction, size, interval, tag = None*8
    with open(csv,'r') as fp:
        for line in fp.readlines():
            ip, port, protocol, direction, dataLen, timeval, tag = line.strip().split(',')
            direction = float(direction)
            if direction == 0:
                traffic.append([float(ip), float(port), float(protocol), 0, 1, float(dataLen)/1000.0, np.float64(timeval), int(tag)])
            else:
                traffic.append([float(ip), float(port), float(protocol), 1, 0, float(dataLen)/1000.0, np.float64(timeval), int(tag)])
    return traffic

def read_from_csv_onlymac(csv):
    traffic = []
    with open(csv, 'r') as fp:
        for line in fp.readlines():
            direction, datalen, timeval, tag = line.strip().split(',')
            direction = float(direction)
            if direction == 0:
                traffic.append([0, 1, float(datalen)/1000.0, np.float64(timeval), int(tag)])
            else:
                traffic.append([1, 0, float(datalen)/1000.0, np.float64(timeval), int(tag)])
    return traffic

def read_from_csv_v3(csv):
    traffic = []
    #org, domain, protocol, port, direction, size, interval, tag = None*8
    with open(csv,'r') as fp:
        for line in fp.readlines():
            ip, port, ipv4, ipv6, tcp, udp, http, ssl, dns, direction, dataLen, timeval, tag = line.strip().split(',')
            direction = float(direction)
            if direction == 0:
                traffic.append([float(ip), float(port), float(ipv4), float(ipv6), float(tcp), float(udp), float(http), float(ssl), float(dns), 0, 1, float(dataLen)/1000.0, np.float64(timeval), int(tag)])
            else:
                traffic.append([float(ip), float(port), float(ipv4), float(ipv6), float(tcp), float(udp), float(http), float(ssl), float(dns), 1, 0, float(dataLen)/1000.0, np.float64(timeval), int(tag)])
    return traffic

def read_from_csv_timeexcluded(csv):
    traffic = []
    #org, domain, protocol, port, direction, size, interval, tag = None*8
    with open(csv,'r') as fp:
        for line in fp.readlines():
            ip, port, protocol, direction, dataLen, timeval, tag = line.strip().split(',')
            direction = float(direction)
            if direction == 0:
                traffic.append([float(ip), float(port), float(protocol), 0, 1, float(dataLen)/1000.0, int(tag)])
            else:
                traffic.append([float(ip), float(port), float(protocol), 1, 0, float(dataLen)/1000.0, int(tag)])
    return traffic

def read_from_csv_timeexcluded_v3(csv):
    traffic = []
    #org, domain, protocol, port, direction, size, interval, tag = None*8
    with open(csv,'r') as fp:
        for line in fp.readlines():
            ip, port, ipv4, ipv6, tcp, udp, http, ssl, dns, direction, dataLen, timeval, tag = line.strip().split(',')
            direction = float(direction)
            if direction == 0:
                traffic.append([float(ip), float(port), float(ipv4), float(ipv6), float(tcp), float(udp), float(http), float(ssl), float(dns), 0, 1, float(dataLen)/1000.0, int(tag)])
            else:
                traffic.append([float(ip), float(port),  float(ipv4), float(ipv6), float(tcp), float(udp), float(http), float(ssl), float(dns), 1, 0, float(dataLen)/1000.0, int(tag)])
    return traffic

def conv_to_ndarray(traffic):
    return np.array(traffic)

def filter_unused(traffic, preserved, timeinterval=0):
    """

    :param traffic: a Numpy.array object
    :param preserved: a list containing ['ip','port','protocol','direction1','direction2','datalen', 'timeval','tag']
    :return: a slicing of traffic
    """
    if timeinterval == 0:
        params = ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','tag']
    else:
        params =  ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','timeinterval','tag']
    mask = [params.index(e) for e in preserved]
    traffic_new = traffic[:,mask]
    return traffic_new

def read_params(path='/home/public/dsk/indexes'):
    with open(path,'r') as fp:
        ipdict = json.loads(fp.readline().strip())
        portdict = json.loads(fp.readline().strip())
        #protdict = json.loads(fp.readline().strip())
    return ipdict, portdict, #protdict

def merge_flows(flow):
    """

    :param flow1: shape ([[],[],...],A)
    :param flow2: shape ([[],[],...],B)
    :return: shape ([[],[],...][A,B]) sorted by timeinterval
    """
    #f = [e[0] for e in flow]
    #l = [e[1] for e in flow]

    f,l = zip(*flow)
    f = np.concatenate(f, axis=0)
    #print 'l', l
    #print 'f.shape', f.shape
    #print 'f', f
    f = f[f[:,-1].argsort()]

    start_time = f[0,-1]
    for i in range(len(f)):
        prev = f[i,-1]
        f[i,-1] -= start_time
        start_time = prev

    return (f, l)

def merge_flows_hard(flow):
    fs,ls = zip(*flow)
    vecs = []
    vecs.extend(fs[0])
    for i in range(len(fs)):
        start_time = fs[i][0, -1]
        for j in range(len(fs[i])):
            prev = fs[i][j,-1]
            fs[i][j,-1] -= start_time
            start_time = prev

    for i in range(1, len(fs)):
        start = 0
        for j in range(len(fs[i])):
            pos = random.randint(start,len(vecs))
            vecs.insert(pos, fs[i][j])
            start = pos
    return (vecs, ls)

def tag_to_idx(tag):
    #tags = [1,14,19,22,25,29,5,10,15,2,23,27,3,6,13,16,21,24]
    tags = range(30)
    return tags.index(tag)

def save_result(filePath, dData):
    """

    :param filePath: The file to write data
    :param dData: dict form data {parameters, trainsample, timeconsume, result}
    :return:
    """
    with open(filePath, 'w') as fp:
        fp.write('algoirithm'+'\t'+str(dData['algorithm'])+'\n')
        fp.write('parameter'+'\t'+str(dData['parameter'])+'\n')
        fp.write('accuracy'+'\t'+str(dData['accuracy'])+'\n')
        fp.write('validating error\n')
        for k in dData['validating error']:
            fp.write(str(k)+'\t'+str(dData['validating error'][k])+' '+str(dData['validating total'][k])+'\n')





