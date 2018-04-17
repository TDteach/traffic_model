import models
from config import DefaultConfig
from data import Dispatcher
from data import FlowExtractor
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
from data import gen_mixture
"""
['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','timeinterval', 'tag']
"""
def train(**kwargs):
    opt = DefaultConfig()
    opt.parse(kwargs)

    #model = models.model_4(41, 16, opt.tag_set_size, 2, opt.use_gpu)
    model = models.CNN(71, 30)
    #print model.ipembed(autograd.Variable(torch.LongTensor([5, 56, 28])))
    if opt.use_gpu:
        model.cuda()
        loss_function = nn.CrossEntropyLoss().cuda()
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)


    train_data = Dispatcher(opt.csv_paths, opt.used_size, opt.capacity, opt.window_size, opt.step, filter=opt.mask, timeinterval=True)
    val_data = Dispatcher(opt.csv_paths, opt.used_size, opt.capacity, opt.window_size, opt.step, test=True, filter=opt.mask, timeinterval=True)
    val_data_2 = Dispatcher(opt.csv_paths, opt.used_size, opt.capacity, opt.window_size/2, opt.step, test=True, filter=opt.mask, timeinterval=True)
    val_data_4 = Dispatcher(opt.csv_paths, opt.used_size, opt.capacity, opt.window_size/4, opt.step, test=True, filter=opt.mask, timeinterval=True)

    train_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=True,drop_last=True, num_workers=opt.num_workers)#,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers,drop_last=True)

    print 'len(train_data) = ', len(train_data)

    stat = dict.fromkeys(range(30), 0)

    for name, param in model.named_parameters():
        print name, param.size()

    for epoch in range(opt.max_epoch):
        print 'epoch = ', epoch

        #loss_meter.reset()
        #confusion_matrix.reset()
        optimizer.zero_grad()

        for i, batch in enumerate(train_dataloader):
            """
            Here we use ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','tag']
            """
            if opt.use_gpu:
                sample_tensor =batch['window'].cuda() #batch['window'].type(torch.cuda.FloatTensor)
            else:
                sample_tensor = batch['window']

            for k in range(len(sample_tensor)):
                start_time = sample_tensor[k][0, -1]
                for j in range(len(sample_tensor[k])):
                    prev = sample_tensor[k][j, -1]
                    sample_tensor[k][j, -1] -= start_time
                    start_time = prev

            batch_size, window_size, feature_len = sample_tensor.size()
            #print 'batch_size, window_size, feature_len', (batch_size, window_size, feature_len)
            inst0 = sample_tensor[:,:,0].contiguous()
            inst1 = sample_tensor[:,:,1].contiguous()
            inst2 = sample_tensor[:,:,2].contiguous()
            inst3 = sample_tensor[:,:,3].contiguous()
            inst4 = sample_tensor[:,:,4].contiguous()
            inst5 = sample_tensor[:,:,5].contiguous()
            inst6 = sample_tensor[:,:,6].contiguous()
            inst7 = sample_tensor[:,:,7].contiguous()
            inst8 = sample_tensor[:,:,8].contiguous()
            inst9 = sample_tensor[:,:,9].contiguous()
            inst10 = sample_tensor[:,:,10].contiguous()
            inst11 = sample_tensor[:,:,11].contiguous()
            inst12 = sample_tensor[:,:,12].contiguous()

            if opt.use_gpu:
                ip_var = autograd.Variable(inst0.type(torch.cuda.LongTensor))
                port_var = autograd.Variable(inst1.type(torch.cuda.LongTensor))
            else:
                ip_var = autograd.Variable(inst0.type(torch.LongTensor))
                port_var = autograd.Variable(inst1.type(torch.LongTensor))

            var_embed = torch.cat((
                model.ipembed(ip_var), #model.ipembed(ip_var2).view(batch_size, window_size, -1),
                model.portembed(port_var),
                autograd.Variable(inst2.view(batch_size, window_size, -1)),
                autograd.Variable(inst3.view(batch_size, window_size, -1)),
                autograd.Variable(inst4.view(batch_size, window_size, -1)),
                autograd.Variable(inst5.view(batch_size, window_size, -1)),
                autograd.Variable(inst6.view(batch_size, window_size, -1)),
                autograd.Variable(inst7.view(batch_size, window_size, -1)),
                autograd.Variable(inst8.view(batch_size, window_size, -1)),
                autograd.Variable(inst9.view(batch_size, window_size, -1)),
                autograd.Variable(inst10.view(batch_size, window_size, -1)),
                autograd.Variable(inst11.view(batch_size, window_size, -1)),
                autograd.Variable(inst12.view(batch_size, window_size, -1)),
                #autograd.Variable(sample_tensor[:, :, 3]).contiguous().view(batch_size, window_size, -1),
                #autograd.Variable(sample_tensor[:, :, 4]).contiguous().view(batch_size, window_size, -1),
                #autograd.Variable(sample_tensor[:, :, 5]).contiguous().view(batch_size, window_size, -1)
            ), 2)
            #print 'var_embed.size()', var_embed.size()
            var_embed = var_embed.view(batch_size, -1, window_size, 71)
            print 'var_embed', var_embed
            if opt.use_gpu:
                label_tensor = torch.from_numpy(conv_to_ndarray([tag_to_idx(e) for e in batch['label']])).cuda()
            else:
                label_tensor = torch.from_numpy(conv_to_ndarray([tag_to_idx(e) for e in batch['label']]))
            """
            Collect the statistic of training procedure
            """
            if epoch == 0:
                for label in label_tensor:
                    stat[label] += 1

            #print 'label_tensor.size()', label_tensor.size()
            #print 'sample_tensor', sample_tensor #size(batchsize*windowsize*feature)
            #print 'labels', batch['label'] #size(batchsize)
            for label in batch['label']:
                if not stat.has_key(label):
                    stat[label] = 1
                else:
                    stat[label] += 1

            model.zero_grad()
            #print 'tag_scores.size()',tag_scores.size() #(batchsize,windowsize,tagsize)
            #print 'last_output.size()',last_output.size() #(batchsize,tagsize)
            #print 'torch.max(last_output,1)[1]', torch.max(last_output,1)[1]
            last_output = model(var_embed)
            #print 'autograd.Variable(label_tensor)[i].size()', autograd.Variable(label_tensor)[i].size()
            #print 'label_tensor.size()', label_tensor.size()
            loss = loss_function(last_output, autograd.Variable(label_tensor))

            loss.backward(retain_graph=True)
            optimizer.step()
            #torch.nn.utils.clip_grad_norm(model.parameters(), True)
            #for p in model.parameters():
            #    p.data.add_(-lr, p.grad.data)

            #loss_meter.add(loss.data[0])
            #confusion_matrix.add(last_output.data, label_tensor)
            #if i % opt.print_freq == opt.print_freq - 1:
            #    vis.plot('loss', loss_meter.value()[0])

    val_cm, val_accuracy = validate(model, val_dataloader)
    validate_mix(model, val_data_2, 2)
    validate_mix(model, val_data_4, 4)

    print 'training procedure'
    for k in stat:
        print 'k = ', k, stat[k]


    return model

def validate_mix(model, val_data, mix=1):
    opt = DefaultConfig()
    correct, false_p, false_n, total = 0, 0, 0, 0
    windows, labels = val_data.windows_, val_data.labels_
    if mix == 0 or mix == 1:
        validate_set = random.sample(zip(windows, labels), opt.capacity)
    else:
        validate_set = gen_mixture(windows, labels, opt.capacity, mix)

    fs, ls = zip(*validate_set)
    model.eval()
    for i in range(len(fs)):
        sample_tensor = fs[i]
        labels = ls[i]
        if type(labels) == np.float64:
            labels = [labels]
        labels = [tag_to_idx(e) for e in labels]

        label_tensor = conv_to_ndarray(labels)
        label_tensor = torch.from_numpy(label_tensor)
        label_tensor = label_tensor.type(torch.LongTensor)
        sample_tensor = conv_to_ndarray(sample_tensor)
        sample_tensor = torch.from_numpy(sample_tensor)
        sample_tensor = sample_tensor.type(torch.FloatTensor)

        if mix == 0 or mix == 1:
            start_time = sample_tensor[0, -1]
            for j in range(len(sample_tensor)):
                prev = sample_tensor[j, -1]
                sample_tensor[j, -1] -= start_time
                start_time = prev
        if opt.use_gpu:
            label_tensor = label_tensor.cuda()
            sample_tensor = sample_tensor.cuda()
        # print 'batch_size, window_size, feature_len', (batch_size, window_size, feature_len)
        sample_length = len(sample_tensor)
        inst0 = sample_tensor[:, 0].contiguous()
        inst1 = sample_tensor[:, 1].contiguous()
        inst2 = sample_tensor[:, 2].contiguous()
        inst3 = sample_tensor[:, 3].contiguous()
        inst4 = sample_tensor[:, 4].contiguous()
        inst5 = sample_tensor[:, 5].contiguous()
        inst6 = sample_tensor[:, 6].contiguous()
        inst7 = sample_tensor[:, 7].contiguous()
        inst8 = sample_tensor[:, 8].contiguous()
        inst9 = sample_tensor[:, 9].contiguous()
        inst10 = sample_tensor[:, 10].contiguous()
        inst11 = sample_tensor[:, 11].contiguous()
        inst12 = sample_tensor[:, 12].contiguous()

        if opt.use_gpu:
            ip_var = autograd.Variable(inst0.type(torch.cuda.LongTensor))
            port_var = autograd.Variable(inst1.type(torch.cuda.LongTensor))
        else:
            ip_var = autograd.Variable(inst0.type(torch.LongTensor))
            port_var = autograd.Variable(inst1.type(torch.LongTensor))

        var_embed = torch.cat((
            model.ipembed(ip_var).view(1, sample_length, -1),  # model.ipembed(ip_var2).view(batch_size, window_size, -1),
            model.portembed(port_var).view(1, sample_length, -1),
            autograd.Variable(inst2.view(1, sample_length, -1)),
            autograd.Variable(inst3.view(1, sample_length, -1)),
            autograd.Variable(inst4.view(1, sample_length, -1)),
            autograd.Variable(inst5.view(1, sample_length, -1)),
            autograd.Variable(inst6.view(1, sample_length, -1)),
            autograd.Variable(inst7.view(1, sample_length, -1)),
            autograd.Variable(inst8.view(1, sample_length, -1)),
            autograd.Variable(inst9.view(1, sample_length, -1)),
            autograd.Variable(inst10.view(1, sample_length, -1)),
            autograd.Variable(inst11.view(1, sample_length, -1)),
            autograd.Variable(inst12.view(1, sample_length, -1)),
        ), 2)
        var_embed = var_embed.view(1, -1, sample_length, 71)

        scores = model(var_embed)
        largest = scores.sort(descending=True)[-1].view(-1)[:mix].data
        m = set(largest) & set(label_tensor)
        p = set(largest) - set(label_tensor)
        n = set(label_tensor) - set(largest)

        correct += len(m)
        false_p += len(p)
        false_n += len(n)
        total += len(label_tensor)


    print "false_positive", 1.0 * false_p / total
    print "false_negative", 1.0 * false_n / total
    print "precise", 1.0 * correct / total



def validate(model, dataloader):
    opt = DefaultConfig()
    total, correct = 0, 0
    error_d = dict.fromkeys(range(30), 0)
    total_d = dict.fromkeys(range(30), 0)

    model.eval()
    #confusion_matrix = meter.ConfusionMeter(30)
    confusion_matrix = CM.ConfusionMatrix(30)
    confusion_matrix.reset()

    for i, data in enumerate(dataloader):
        input, label = data['window'], data['label']
        val_input = input.type(torch.FloatTensor)
        val_label = autograd.Variable(torch.LongTensor([tag_to_idx(e) for e in data['label']]))

        for k in range(len(val_input)):
            start_time = val_input[k][0, -1]
            for j in range(len(val_input[k])):
                prev = val_input[k][j, -1]
                val_input[k][j, -1] -= start_time
                start_time = prev

        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()

        batch_size, window_size, feature_len = val_input.size()
        inst0 = val_input[:, :, 0].contiguous()
        inst1 = val_input[:, :, 1].contiguous()
        inst2 = val_input[:, :, 2].contiguous()
        inst3 = val_input[:, :, 3].contiguous()
        inst4 = val_input[:, :, 4].contiguous()
        inst5 = val_input[:, :, 5].contiguous()
        inst6 = val_input[:, :, 6].contiguous()
        inst7 = val_input[:, :, 7].contiguous()
        inst8 = val_input[:, :, 8].contiguous()
        inst9 = val_input[:, :, 9].contiguous()
        inst10 = val_input[:, :, 10].contiguous()
        inst11 = val_input[:, :, 11].contiguous()
        inst12 = val_input[:, :, 12].contiguous()

        if opt.use_gpu:
            ip_var = autograd.Variable(inst0.type(torch.cuda.LongTensor))
            port_var = autograd.Variable(inst1.type(torch.cuda.LongTensor))
        else:
            ip_var = autograd.Variable(inst0.type(torch.LongTensor))
            port_var = autograd.Variable(inst1.type(torch.LongTensor))

        var_embed = torch.cat((
            model.ipembed(ip_var),  # model.ipembed(ip_var2).view(batch_size, window_size, -1),
            model.portembed(port_var),
            autograd.Variable(inst2.view(batch_size, window_size, -1)),
            autograd.Variable(inst3.view(batch_size, window_size, -1)),
            autograd.Variable(inst4.view(batch_size, window_size, -1)),
            autograd.Variable(inst5.view(batch_size, window_size, -1)),
            autograd.Variable(inst6.view(batch_size, window_size, -1)),
            autograd.Variable(inst7.view(batch_size, window_size, -1)),
            autograd.Variable(inst8.view(batch_size, window_size, -1)),
            autograd.Variable(inst9.view(batch_size, window_size, -1)),
            autograd.Variable(inst10.view(batch_size, window_size, -1)),
            autograd.Variable(inst11.view(batch_size, window_size, -1)),
            autograd.Variable(inst12.view(batch_size, window_size, -1)),
            # autograd.Variable(sample_tensor[:, :, 3]).contiguous().view(batch_size, window_size, -1),
            # autograd.Variable(sample_tensor[:, :, 4]).contiguous().view(batch_size, window_size, -1),
            # autograd.Variable(sample_tensor[:, :, 5]).contiguous().view(batch_size, window_size, -1)
        ), 2)
        var_embed = var_embed.view(batch_size, -1, window_size, 71)
        last_output = model(var_embed)
        #confusion_matrix.add(last_output.data.squeeze(), val_label.data.type(torch.LongTensor))

        _, predicted = torch.max(last_output.data, 1)
        total += val_label.size()[0]
        correct += (predicted == val_label.data).sum()

        for j in range(val_label.size()[0]):
            confusion_matrix.add(predicted[j], val_label.data[j])

        """
        Some operations on my confusion matrix... 
        """
        for e in val_label.data:
            total_d[e] += 1
        error = val_label.data[val_label.data != predicted]
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


