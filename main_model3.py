import models
from config import DefaultConfig
from data import Dispatcher
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

import time


def train(**kwargs):
    opt = DefaultConfig()
    opt.parse(kwargs)

    model = models.model_1(13, 16, opt.tag_set_size, 2, opt.use_gpu)
    #print model.ipembed(autograd.Variable(torch.LongTensor([5, 56, 28])))
    if opt.use_gpu:
        model.cuda()
        loss_function = nn.CrossEntropyLoss().cuda()
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    train_data = Dispatcher(opt.csv_paths, opt.used_size, opt.window_size, opt.step, filter=opt.mask)
    val_data = Dispatcher(opt.csv_paths, opt.used_size, opt.window_size, opt.step, test=True, filter=opt.mask)
    train_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=True,drop_last=True, num_workers=opt.num_workers)#,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers,drop_last=True)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    stat = dict.fromkeys(range(30), 0)

    for name, param in model.named_parameters():
        print name, param.size()

    for epoch in range(opt.max_epoch):
        print 'epoch = ', epoch
        loss_meter.reset()
        confusion_matrix.reset()
        optimizer.zero_grad()

        for i, batch in enumerate(train_dataloader):
            """
            Here we use ['protocol','direction1','direction2','datalen','tag']
            """
            sample_tensor =batch['window'].cuda() #batch['window'].type(torch.cuda.FloatTensor)
            batch_size, window_size, feature_len = sample_tensor.size()

            inst0 = sample_tensor[:,:,0].contiguous()
            inst1 = sample_tensor[:,:,1].contiguous()
            inst2 = sample_tensor[:,:,2].contiguous()
            inst3 = sample_tensor[:,:,3].contiguous()

            prot_var = autograd.Variable(inst0.type(torch.cuda.LongTensor))

            var_embed = torch.cat((
                model.protembed(prot_var), #model.ipembed(ip_var2).view(batch_size, window_size, -1),
                autograd.Variable(inst1.view(batch_size, window_size, -1)),
                autograd.Variable(inst2.view(batch_size, window_size, -1)),
                autograd.Variable(inst3.view(batch_size, window_size, -1))
                #autograd.Variable(sample_tensor[:, :, 3]).contiguous().view(batch_size, window_size, -1),
                #autograd.Variable(sample_tensor[:, :, 4]).contiguous().view(batch_size, window_size, -1),
                #autograd.Variable(sample_tensor[:, :, 5]).contiguous().view(batch_size, window_size, -1)
            ), 2)
            #print 'var_embed', var_embed

            label_tensor = torch.from_numpy(conv_to_ndarray([tag_to_idx(e) for e in batch['label']])).cuda()
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
            loss = loss_function(last_output, autograd.Variable(label_tensor))

            loss.backward(retain_graph=True)
            optimizer.step()
            #torch.nn.utils.clip_grad_norm(model.parameters(), True)
            #for p in model.parameters():
            #    p.data.add_(-lr, p.grad.data)
    model.save()
    print 'training procedure'
    for k in stat:
        print 'k = ', k, stat[k]

    validate(model, val_dataloader)

    return model

def validate(model, dataloader):
    opt = DefaultConfig()
    total, correct = 0, 0
    error_d = dict.fromkeys(range(30), 0)
    total_d = dict.fromkeys(range(30), 0)

    model.eval()
    for i, data in enumerate(dataloader):
        input, label = data['window'], data['label']
        val_input = input.type(torch.FloatTensor)
        val_label = autograd.Variable(torch.LongTensor([tag_to_idx(e) for e in data['label']]))

        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()

        batch_size, window_size, feature_len = val_input.size()
        inst0 = val_input[:, :, 0].contiguous()
        inst1 = val_input[:, :, 1].contiguous()
        inst2 = val_input[:, :, 2].contiguous()
        inst3 = val_input[:, :, 3].contiguous()

        prot_var = autograd.Variable(inst0.type(torch.cuda.LongTensor))
        val_embed = torch.cat((
                model.protembed(prot_var), #model.ipembed(ip_var2).view(batch_size, window_size, -1),
                autograd.Variable(inst1.view(batch_size, window_size, -1)),
                autograd.Variable(inst2.view(batch_size, window_size, -1)),
                autograd.Variable(inst3.view(batch_size, window_size, -1))
            ), 2)
        last_output = model(val_embed)
        #print 'score = ', score
        _, predicted = torch.max(last_output.data, 1)
        #print 'predicted = ', predicted
        #print 'val_label.data = ', val_label.data
        total += val_label.size()[0]
        correct += (predicted == val_label.data).sum()

        for e in val_label.data:
            total_d[e] += 1
        error = val_label.data[val_label.data != predicted]
        for e in error:
            error_d[e] += 1


    acc_rate = 1.0 * correct / total
    print 'accuracy = ', acc_rate
    for k in range(30):
        print 'k = ', k, error_d[k], total_d[k]
    keys = range(30)
    dData = {'parameter':str(opt.max_epoch)+'|'+str(opt.lr)+'|'+str(opt.window_size)+'|'+','.join(opt.mask),
             'validating total':{k:total_d[k] for k in keys},
             'validating error':{k:error_d[k] for k in keys},
             'accuracy':acc_rate}
    save_path = '/home/4tshare/iot/dev/traffic_model/results/' + time.strftime('%m%d_%H:%M:%S')
    save_result(save_path, dData)

    """
    for images, labels in test_loader:
    images = Variable(images.view(-1, sequence_length, input_size))
    outputs = rnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    """

if __name__ == "__main__":
    print 'torch.__version__', torch.__version__
    model = train()


