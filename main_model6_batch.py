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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','tag']
"""
"""
In this module, we add timeinterval into the model.
"""
def train(**kwargs):
    opt = DefaultConfig()
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    model = models.model_4(71, 64, opt.tag_set_size, 2, opt.use_gpu)
    #print model.ipembed(autograd.Variable(torch.LongTensor([5, 56, 28])))
    if opt.use_gpu:
        model.cuda()
        loss_function = nn.CrossEntropyLoss().cuda()
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    lr = opt.lr

    """
    train_data = Dispatcher(opt.csv_paths, opt.used_size, opt.window_size, opt.step, filter=opt.mask)
    val_data = Dispatcher(opt.csv_paths, opt.used_size, opt.window_size, opt.step, test=True, filter=opt.mask)
    train_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=True,drop_last=True, num_workers=opt.num_workers)#,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers,drop_last=True)
    """
    train_data = FlowExtractor_time(opt.csv_paths, opt.used_size, opt.capacity, opt.window_size, opt.step, filter=opt.mask)
    val_data = FlowExtractor_time(opt.csv_paths, opt.used_size, opt.capacity , opt.window_size, opt.step, test=True, filter=opt.mask)
    val_data_2 = FlowExtractor_time(opt.csv_paths, opt.used_size, 2, opt.capacity, opt.window_size, opt.step, test=True, filter=opt.mask)
    val_data_4 = FlowExtractor_time(opt.csv_paths, opt.used_size, 2, opt.capacity, opt.window_size, opt.step, test=True, filter=opt.mask)
    print 'len(train_data) = ', len(train_data)

    stat = dict.fromkeys(range(30), 0)

    for name, param in model.named_parameters():
        print name, param.size()

    for epoch in range(opt.max_epoch):
        print 'epoch = ', epoch

        #loss_meter.reset()
        #confusion_matrix.reset()
        optimizer.zero_grad()

        #shuffle the windows and labels every epoch.
        permutation = np.random.permutation(len(train_data))
        train_data.windows_ = train_data.windows_[permutation]
        train_data.labels_ = train_data.labels_[permutation]

        for i in range(len(train_data)/opt.batch_size):
            """
            Here we use ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','tag']
            """
            sample_tensor = [elem for elem in train_data.windows_[i:i+opt.batch_size]]
            label_tensor = [elem for elem in train_data.labels_[i:i+opt.batch_size]]
            """
            sample_tensor shape: [ [[ip,port,...,time],[ip,port,...,time],...,[ip,port,...,time]],
                                   [[ip,port,...,time],[ip,port,...,time],...,[ip,port,...,time]],
                                     ......
                                 ]
            """

            for k in range(len(sample_tensor)):
                start_time = sample_tensor[k][0,-1]
                for j in range(len(sample_tensor[k])):
                    prev = sample_tensor[k][j, -1]
                    sample_tensor[k][j,-1] -= start_time
                    start_time = prev

            sample_length = torch.LongTensor([x for x in map(len, sample_tensor)])
            seq_tensor = torch.zeros((len(sample_tensor), sample_length.max(), 13))
            for idx, (seq, seqlen) in enumerate(zip(sample_tensor, sample_length)):
                seq_tensor[idx, :seqlen, :] =  torch.FloatTensor(seq)
            sample_length, perm_idx = sample_length.sort(0, True)
            seq_tensor = seq_tensor[perm_idx]
            label_tensor = conv_to_ndarray(label_tensor)
            label_tensor = label_tensor[perm_idx]
            label_tensor = torch.from_numpy(conv_to_ndarray([tag_to_idx(e) for e in label_tensor]))

            #print 'sample_length', sample_length

            # If the dimension of inner vectors is not consistent. torch.Tensor.size() can only return the most external dimension
            #sample_tensor =batch['window'].cuda() #batch['window'].type(torch.cuda.FloatTensor)
            #batch_size, window_size, feature_len = sample_tensor.size()

            if opt.use_gpu:
                seq_tensor = seq_tensor.cuda()
                label_tensor = label_tensor.cuda()

            inst0 = seq_tensor[:,:,0].contiguous()
            inst1 = seq_tensor[:,:,1].contiguous()
            inst2 = seq_tensor[:,:,2].contiguous()
            inst3 = seq_tensor[:,:,3].contiguous()
            inst4 = seq_tensor[:,:,4].contiguous()
            inst5 = seq_tensor[:,:,5].contiguous()
            inst6 = seq_tensor[:,:,6].contiguous()
            inst7 = seq_tensor[:,:,7].contiguous()
            inst8 = seq_tensor[:,:,8].contiguous()
            inst9 = seq_tensor[:,:,9].contiguous()
            inst10 = seq_tensor[:,:,10].contiguous()
            inst11 = seq_tensor[:,:,11].contiguous()
            inst12 = seq_tensor[:,:,12].contiguous()

            ip_var = autograd.Variable(inst0.type(torch.cuda.LongTensor))
            port_var = autograd.Variable(inst1.type(torch.cuda.LongTensor))

            var_embed = torch.cat((
                model.ipembed(ip_var), #model.ipembed(ip_var2).view(batch_size, window_size, -1),
                model.portembed(port_var),
                autograd.Variable(inst2.view(opt.batch_size, sample_length.max(), -1)),
                autograd.Variable(inst3.view(opt.batch_size, sample_length.max(), -1)),
                autograd.Variable(inst4.view(opt.batch_size, sample_length.max(), -1)),
                autograd.Variable(inst5.view(opt.batch_size, sample_length.max(), -1)),
                autograd.Variable(inst6.view(opt.batch_size, sample_length.max(), -1)),
                autograd.Variable(inst7.view(opt.batch_size, sample_length.max(), -1)),
                autograd.Variable(inst8.view(opt.batch_size, sample_length.max(), -1)),
                autograd.Variable(inst9.view(opt.batch_size, sample_length.max(), -1)),
                autograd.Variable(inst10.view(opt.batch_size, sample_length.max(), -1)),
                autograd.Variable(inst11.view(opt.batch_size, sample_length.max(), -1)),
                autograd.Variable(inst12.view(opt.batch_size, sample_length.max(), -1)),
            ), 2)
            packed_input = pack_padded_sequence(var_embed, sample_length.numpy(), batch_first=True)
            #print 'packed_input', packed_input
            #print 'var_embed', var_embed

            if opt.use_gpu:
                packed_input = packed_input.cuda()

            """
            Collect the statistic of training procedure
            """
            if epoch == 0:
                for label in label_tensor:
                    stat[label] += 1

            #print 'label_tensor.size()', label_tensor.size()
            #print 'sample_tensor', sample_tensor #size(batchsize*windowsize*feature)
            #print 'labels', batch['label'] #size(batchsize)
            for label in label_tensor:
                if not stat.has_key(label):
                    stat[label] = 1
                else:
                    stat[label] += 1

            model.zero_grad()
            #print 'tag_scores.size()',tag_scores.size() #(batchsize,windowsize,tagsize)
            #print 'last_output.size()',last_output.size() #(batchsize,tagsize)
            #print 'torch.max(last_output,1)[1]', torch.max(last_output,1)[1]
            last_output = model(packed_input, opt.batch_size)
            #print 'last_output', last_output
            #print 'autograd.Variable(label_tensor)[i].size()', autograd.Variable(label_tensor)[i].size()
            loss = loss_function(last_output, autograd.Variable(label_tensor))

            loss.backward(retain_graph=True)
            optimizer.step()

        model.save()

    val_cm, val_accuracy = validate(model, val_data)

    print 'training procedure'
    for k in stat:
        print 'k = ', k, stat[k]

    return model

def validate(model, test_data):
    opt = DefaultConfig()
    total, correct = 0, 0
    error_d = dict.fromkeys(range(30), 0)
    total_d = dict.fromkeys(range(30), 0)

    model.eval()
    #confusion_matrix = meter.ConfusionMeter(30)
    confusion_matrix = CM.ConfusionMatrix(30)
    confusion_matrix.reset()

    for i in range(len(test_data) / opt.batch_size):
        """
        Here we use ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','tag']
        """
        sample_tensor = [elem for elem in test_data.windows_[i:i + opt.batch_size]]
        label_tensor = [elem for elem in test_data.labels_[i:i + opt.batch_size]]
        """
        sample_tensor shape: [ [[ip,port,...,time],[ip,port,...,time],...,[ip,port,...,time]],
                               [[ip,port,...,time],[ip,port,...,time],...,[ip,port,...,time]],
                                 ......
                             ]
        """

        for k in range(len(sample_tensor)):
            start_time = sample_tensor[k][0, -1]
            for j in range(len(sample_tensor[k])):
                prev = sample_tensor[k][j, -1]
                sample_tensor[k][j, -1] -= start_time
                start_time = prev

        sample_length = torch.LongTensor([x for x in map(len, sample_tensor)])
        seq_tensor = torch.zeros((len(sample_tensor), sample_length.max(), 13))
        for idx, (seq, seqlen) in enumerate(zip(sample_tensor, sample_length)):
            seq_tensor[idx, :seqlen, :] = torch.FloatTensor(seq)
        sample_length, perm_idx = sample_length.sort(0, True)
        label_tensor = conv_to_ndarray(label_tensor)
        seq_tensor = seq_tensor[perm_idx]
        label_tensor = label_tensor[perm_idx]
        label_tensor = torch.from_numpy(conv_to_ndarray([tag_to_idx(e) for e in label_tensor]))

        if opt.use_gpu:
            seq_tensor = seq_tensor.cuda()
            label_tensor = label_tensor.cuda()

        inst0 = seq_tensor[:, :, 0].contiguous()
        inst1 = seq_tensor[:, :, 1].contiguous()
        inst2 = seq_tensor[:, :, 2].contiguous()
        inst3 = seq_tensor[:, :, 3].contiguous()
        inst4 = seq_tensor[:, :, 4].contiguous()
        inst5 = seq_tensor[:, :, 5].contiguous()
        inst6 = seq_tensor[:, :, 6].contiguous()
        inst7 = seq_tensor[:, :, 7].contiguous()
        inst8 = seq_tensor[:, :, 8].contiguous()
        inst9 = seq_tensor[:, :, 9].contiguous()
        inst10 = seq_tensor[:, :, 10].contiguous()
        inst11 = seq_tensor[:, :, 11].contiguous()
        inst12 = seq_tensor[:, :, 12].contiguous()

        ip_var = autograd.Variable(inst0.type(torch.cuda.LongTensor))
        port_var = autograd.Variable(inst1.type(torch.cuda.LongTensor))

        var_embed = torch.cat((
            model.ipembed(ip_var),  # model.ipembed(ip_var2).view(batch_size, window_size, -1),
            model.portembed(port_var),
            autograd.Variable(inst2.view(opt.batch_size, sample_length.max(), -1)),
            autograd.Variable(inst3.view(opt.batch_size, sample_length.max(), -1)),
            autograd.Variable(inst4.view(opt.batch_size, sample_length.max(), -1)),
            autograd.Variable(inst5.view(opt.batch_size, sample_length.max(), -1)),
            autograd.Variable(inst6.view(opt.batch_size, sample_length.max(), -1)),
            autograd.Variable(inst7.view(opt.batch_size, sample_length.max(), -1)),
            autograd.Variable(inst8.view(opt.batch_size, sample_length.max(), -1)),
            autograd.Variable(inst9.view(opt.batch_size, sample_length.max(), -1)),
            autograd.Variable(inst10.view(opt.batch_size, sample_length.max(), -1)),
            autograd.Variable(inst11.view(opt.batch_size, sample_length.max(), -1)),
            autograd.Variable(inst12.view(opt.batch_size, sample_length.max(), -1)),
            # autograd.Variable(sample_tensor[:, :, 3]).contiguous().view(batch_size, window_size, -1),
            # autograd.Variable(sample_tensor[:, :, 4]).contiguous().view(batch_size, window_size, -1),
            # autograd.Variable(sample_tensor[:, :, 5]).contiguous().view(batch_size, window_size, -1)
        ), 2)
        packed_input = pack_padded_sequence(var_embed, sample_length.numpy(), batch_first=True)
        # print 'var_embed', var_embed

        if opt.use_gpu:
            packed_input.cuda()
        #confusion_matrix.add(last_output.data.squeeze(), val_label.data.type(torch.LongTensor))

        last_output = model(packed_input, opt.batch_size)

        _, predicted = torch.max(last_output.data, 1)
        total += len(sample_length)
        correct += (predicted == label_tensor).sum()

        for j in range(label_tensor.size()[0]):
            confusion_matrix.add(predicted[j], label_tensor[j])

        """
        Some operations on my confusion matrix... 
        """
        for e in label_tensor:
            total_d[e] += 1
        error = label_tensor[label_tensor != predicted]
        for e in error:
            error_d[e] += 1


    model.train()
    acc_rate = 1.0 * correct / total
    print 'accuracy = ', acc_rate
    for k in range(30):
        print 'k = ', k, error_d[k], total_d[k]
    keys = range(30)
    dData = {'parameter':str(opt.max_epoch)+'|'+str(opt.lr)+'|'+str(opt.window_size)+'|'+','.join(opt.mask),
             'algorithm': 'LSTM',
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
    model = train()


