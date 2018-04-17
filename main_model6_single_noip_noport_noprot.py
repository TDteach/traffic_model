import models
from config import DefaultConfig
from data import FlowExtractor
from data import FlowExtractor_time
from utils import tag_to_idx
from utils import conv_to_ndarray
from utils import save_result
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchnet import meter
from utils.visualize import Visualizer
import os
import numpy as np
from utils import confusion_matrix as CM
from utils import read_from_csv_v3
import time
import random
from torch.nn.utils.rnn import pack_padded_sequence
from data import gen_mixture
"""
['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','tag']
"""
"""
In this module, we add timeinterval into the model.
"""
def trans_embed(model, sample_tensor, labels):
    """

    :param lstm:
    :param var:
    :return: return a packed variable
    """
    opt = DefaultConfig()
    sample_length = torch.LongTensor([x for x in map(len, sample_tensor)])
    seq_tensor = torch.zeros((len(sample_tensor), sample_length.max(), 4))
    for idx, (seq, seqlen) in enumerate(zip(sample_tensor, sample_length)):
        seq_tensor[idx, :seqlen, :] = torch.FloatTensor(seq)
    sample_length, perm_idx = sample_length.sort(0, True)
    seq_tensor = seq_tensor[perm_idx]
    label_tensor = conv_to_ndarray(labels)
    label_tensor = label_tensor[perm_idx]
    label_tensor = torch.from_numpy(conv_to_ndarray([tag_to_idx(e) for e in label_tensor]))

    if opt.use_gpu:
        label_tensor = label_tensor.cuda()
        seq_tensor = seq_tensor.cuda()

    inst0 = seq_tensor[:, :, 0].contiguous()
    inst1 = seq_tensor[:, :, 1].contiguous()
    inst2 = seq_tensor[:, :, 2].contiguous()
    inst3 = seq_tensor[:, :, 3].contiguous()
    #inst4 = seq_tensor[:, :, 4].contiguous()
    #inst5 = seq_tensor[:, :, 5].contiguous()
    #inst6 = seq_tensor[:, :, 6].contiguous()
    #inst7 = seq_tensor[:, :, 7].contiguous()
    #inst8 = seq_tensor[:, :, 8].contiguous()
    #inst9 = seq_tensor[:, :, 9].contiguous()
    #inst10 = seq_tensor[:, :, 10].contiguous()
    #inst11 = seq_tensor[:, :, 11].contiguous()
    #inst12 = seq_tensor[:, :, 12].contiguous()

    #ip_var = autograd.Variable(inst0.type(torch.cuda.LongTensor))
    #port_var = autograd.Variable(inst1.type(torch.cuda.LongTensor))

    port_var = autograd.Variable(inst0.type(torch.cuda.LongTensor))

    var_embed = torch.cat((
        #model.ipembed(ip_var),  # model.ipembed(ip_var2).view(batch_size, window_size, -1),
        #model.portembed(port_var),
        autograd.Variable(inst0.view(opt.batch_size, sample_length.max(), -1)),
        autograd.Variable(inst1.view(opt.batch_size, sample_length.max(), -1)),
        autograd.Variable(inst2.view(opt.batch_size, sample_length.max(), -1)),
        autograd.Variable(inst3.view(opt.batch_size, sample_length.max(), -1)),
        #autograd.Variable(inst4.view(opt.batch_size, sample_length.max(), -1)),
        #autograd.Variable(inst5.view(opt.batch_size, sample_length.max(), -1)),
        #autograd.Variable(inst6.view(opt.batch_size, sample_length.max(), -1)),
        #autograd.Variable(inst7.view(opt.batch_size, sample_length.max(), -1)),
        #autograd.Variable(inst8.view(opt.batch_size, sample_length.max(), -1)),
        #autograd.Variable(inst9.view(opt.batch_size, sample_length.max(), -1)),
        #autograd.Variable(inst10.view(opt.batch_size, sample_length.max(), -1)),
        #autograd.Variable(inst11.view(opt.batch_size, sample_length.max(), -1)),
        #autograd.Variable(inst12.view(opt.batch_size, sample_length.max(), -1)),
    ), 2)
    packed_input = pack_padded_sequence(var_embed, sample_length.numpy(), batch_first=True)
    if opt.use_gpu:
        packed_input = packed_input.cuda()
        label_tensor = label_tensor.cuda()

    return (packed_input, label_tensor)

def train_single(model, training_set, label, capacity):
    opt = DefaultConfig()
    """

    :param lstm: the neural network model
    :param training_set: zip format [([[ip,port,...,],[ip,port,...,],...],0),([[ip,port,...,],[ip,port,...,],...],1)]
    :param label:
    :param capacity:
    :return:
    """
    if opt.use_gpu:
        model.cuda()
        loss_function = nn.CrossEntropyLoss().cuda()
    else:
        loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    pos_set = [list(elem) for elem in training_set if elem[-1] == label]
    neg_set = [list(elem) for elem in training_set if elem[-1] != label]

    if len(pos_set) == 0:
        return None

    pos_set = np.array(random.sample(pos_set, min(capacity, len(pos_set))))
    neg_set = np.array(random.sample(neg_set, min(capacity, len(neg_set))))
    print 'label', label, 'pos', len(pos_set), 'neg',len(neg_set)

    for elem in pos_set:
        elem[-1] = 1
    for elem in neg_set:
        elem[-1] = 0
    comb = np.concatenate((pos_set, neg_set), axis=0)
    np.random.shuffle(comb)

    windows, labels = comb[:, 0], comb[:, -1]
    for epoch in range(opt.max_epoch):
        print 'label = ', label, 'epoch = ', epoch
        permutation = np.random.permutation(len(windows))
        windows = windows[permutation]
        labels = labels[permutation]
        for k in range(len(windows)/opt.batch_size):
            sample_tensor = [elem for elem in windows[k: k+opt.batch_size]]
            label_tensor = [elem for elem in labels[k: k+opt.batch_size]]

            for i in range(len(sample_tensor)):
                start_time = sample_tensor[i][0,-1]
                for j in range(len(sample_tensor[i])):
                    prev = sample_tensor[i][j, -1]
                    sample_tensor[i][j,-1] -= start_time
                    start_time = prev

            packed_input,  label_tensor = trans_embed(model, sample_tensor, label_tensor)
            #print 'packed_input', packed_input
            if opt.use_gpu:
                packed_input = packed_input.cuda()
                label_tensor = label_tensor.cuda()

            model.zero_grad()
            last_output = model(packed_input, opt.batch_size)
            loss = loss_function(last_output, autograd.Variable(label_tensor))
            loss.backward(retain_graph=True)
            optimizer.step()
    return model


def train(**kwargs):
    opt = DefaultConfig()
    opt.parse(kwargs)

    model_vec = [models.model_4(4, 16, 2, 8, opt.use_gpu) for i in range(30)]
    """
    train_data = Dispatcher(opt.csv_paths, opt.used_size, opt.window_size, opt.step, filter=opt.mask)
    val_data = Dispatcher(opt.csv_paths, opt.used_size, opt.window_size, opt.step, test=True, filter=opt.mask)
    train_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=True,drop_last=True, num_workers=opt.num_workers)#,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers,drop_last=True)
    """
    train_data = FlowExtractor_time(opt.csv_paths, opt.used_size, opt.capacity, opt.window_size, opt.step, filter=opt.mask1)
    val_data = FlowExtractor_time(opt.csv_paths, opt.used_size, opt.capacity , opt.window_size, opt.step, test=True, filter=opt.mask1)

    print 'len(train_data) = ', len(train_data)

    training_set = zip(train_data.windows, train_data.labels)
    for i in range(30):
        model_vec[i] = train_single(model_vec[i], training_set, i, opt.capacity)


    validate(model_vec, train_data, 0)
    validate(model_vec, val_data, 2)
    validate(model_vec, val_data, 4)

    return model_vec

def validate_single(model, seq_tensor):
    opt = DefaultConfig()
    if opt.use_gpu:
        model.cuda()

    """start_time = seq_tensor[0, -1]
    for j in range(len(seq_tensor)):
        prev = seq_tensor[j, -1]
        seq_tensor[j, -1] -= start_time
        start_time = prev
    """

    sample_length = len(seq_tensor)
    seq_tensor = torch.from_numpy(seq_tensor)
    if opt.use_gpu:
        seq_tensor = seq_tensor.type(torch.cuda.FloatTensor)
    else:
        seq_tensor = seq_tensor.type(torch.FloatTensor)
    #print 'seq_tensor', seq_tensor

    inst0 = seq_tensor[:, 0].contiguous()
    inst1 = seq_tensor[:, 1].contiguous()
    inst2 = seq_tensor[:, 2].contiguous()
    inst3 = seq_tensor[:, 3].contiguous()
    inst4 = seq_tensor[:, 4].contiguous()
    #inst5 = seq_tensor[:, 5].contiguous()
    #inst6 = seq_tensor[:, 6].contiguous()
    #inst7 = seq_tensor[:, 7].contiguous()
    #inst8 = seq_tensor[:, 8].contiguous()
    #inst9 = seq_tensor[:, 9].contiguous()
    #inst10 = seq_tensor[:, 10].contiguous()
    #inst11 = seq_tensor[:, 11].contiguous()
    #inst12 = seq_tensor[:, 12].contiguous()

    #ip_var = autograd.Variable(inst0.type(torch.cuda.LongTensor))
    port_var = autograd.Variable(inst1.type(torch.cuda.LongTensor))

    #print 'ip_var', ip_var
    #print 'model.ipembed(ip_var)', model.ipembed(ip_var)
    var_embed = torch.cat((
        #model.ipembed(ip_var).view(1, sample_length, -1),  # model.ipembed(ip_var2).view(batch_size, window_size, -1),
        #model.portembed(port_var).view(1, sample_length, -1),
        autograd.Variable(inst0.view(1, sample_length, -1)),
        autograd.Variable(inst1.view(1, sample_length, -1)),
        autograd.Variable(inst2.view(1, sample_length, -1)),
        autograd.Variable(inst3.view(1, sample_length, -1)),
        autograd.Variable(inst4.view(1, sample_length, -1)),
        #autograd.Variable(inst5.view(1, sample_length, -1)),
        #autograd.Variable(inst6.view(1, sample_length, -1)),
        #autograd.Variable(inst7.view(1, sample_length, -1)),
        #autograd.Variable(inst8.view(1, sample_length, -1)),
        #autograd.Variable(inst9.view(1, sample_length, -1)),
        #autograd.Variable(inst10.view(1, sample_length, -1)),
        #autograd.Variable(inst11.view(1, sample_length, -1)),
        #autograd.Variable(inst12.view(1, sample_length, -1)),
    ), 2)
    packed_input = pack_padded_sequence(var_embed, [sample_length], batch_first=1)
    if opt.use_gpu:
        var_embed = var_embed.cuda()
        packed_input = packed_input.cuda()


    result = torch.max(model(packed_input, 1),1)[1]
    #print 'result', result
    return result

def validate(model_vec, flowextractor, mix=1):
    opt = DefaultConfig()
    correct, false_p, false_n, total = 0, 0, 0, 0

    windows, labels = flowextractor.windows_, flowextractor.labels_
    if mix == 0 or mix == 1:
        validate_set = random.sample(zip(windows, labels), opt.capacity)
    else:
        validate_set = gen_mixture(windows, labels, opt.capacity, mix)
    """
    gen_mixture has already calculate the difference of timestamp
    """
    fs, ls = zip(*validate_set)

    for i in range(len(model_vec)):
        if model_vec[i]:
            model_vec[i].eval()

    for i in range(len(fs)):
        seq_tensor = fs[i]
        labels = ls[i]

        if type(labels) == np.float64:
            labels = [labels]

        if mix == 0:
            start_time = seq_tensor[0, -1]
            for j in range(len(seq_tensor)):
                prev = seq_tensor[j, -1]
                seq_tensor[j, -1] -= start_time
                start_time = prev

        #print 'seq_tensor', seq_tensor
        #tmp_res = set([k for k in range(len(model_vec)) if model_vec[k] and validate_single(model_vec[k], seq_tensor) == 1])
        tmp_res = []
        for k in range(len(model_vec)):
            if model_vec[k]:
                if validate_single(model_vec[k], seq_tensor) == 1:
                    tmp_res.append(k)

        m = set(tmp_res) & set(labels)
        p = set(tmp_res) - set(labels)
        n = set(labels) - set(tmp_res)

        correct += len(m)
        false_p += len(p)
        false_n += len(n)
        total += len(labels)

    print "false_positive", 1.0 * false_p / total
    print "false_negative", 1.0 * false_n / total
    print "precise", 1.0 * correct / total



if __name__ == "__main__":
    print 'torch.__version__', torch.__version__
    model_vec = train()


