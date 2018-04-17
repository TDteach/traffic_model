from ..data import trafficReader
from ..data import trafficContainer
import random
import numpy as np
from sklearn.model_selection import train_test_split
from ..statistic import genStatisVec
from ..statistic import genStatisVec2
from ..utils import conv_to_ndarray, tag_to_idx
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.externals import joblib
from ..data import gen_mixture
from ..data import gen_hardmix
from torch.utils.data import DataLoader
from ..config import DefaultConfig
import torch
import torch.autograd as autograd
import torch.nn as nn
from ..models import CNN
from ..models import LSTM
from ..models import LinearLSTM
from ..models import CNN2
from ..models import CNN_LSTM
from ..models import CNN_LSTM2
from multiprocessing import Process
import torchvision
from torchvision import datasets, models, transforms
import os

fea_len = 71

def genDataset(readers, capacity, windowsize, step, test_size=0.2):
    dataset, trainset, testset = [], [], []
    stat = {}.fromkeys(range(30), 0)
    for reader in readers:
        reader.removeWindowsLabels()
        reader.genWindows(capacity, windowsize, step)
        dataset.extend(reader.getWindowsLabels())
    X, y = zip(*dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=10)
    for e in y:
        stat[e] += 1
    print 'stats of devices'
    for e in stat:
        print 'dev = ', e, stat[e]
    return X_train, X_test, y_train, y_test

def trans_embed(model, window, labels):
    opt = DefaultConfig()
    sample_tensor = window
    batch_size, window_size, feature_len = sample_tensor.size()
    if opt.use_gpu:
        sample_tensor = sample_tensor.cuda()
    # print 'batch_size, window_size, feature_len', (batch_size, window_size, feature_len)
    inst0 = sample_tensor[:, :, 0].contiguous()
    inst1 = sample_tensor[:, :, 1].contiguous()
    inst2 = sample_tensor[:, :, 2].contiguous()
    inst3 = sample_tensor[:, :, 3].contiguous()
    inst4 = sample_tensor[:, :, 4].contiguous()
    inst5 = sample_tensor[:, :, 5].contiguous()
    inst6 = sample_tensor[:, :, 6].contiguous()
    inst7 = sample_tensor[:, :, 7].contiguous()
    inst8 = sample_tensor[:, :, 8].contiguous()
    inst9 = sample_tensor[:, :, 9].contiguous()
    inst10 = sample_tensor[:, :, 10].contiguous()
    inst11 = sample_tensor[:, :, 11].contiguous()
    inst12 = sample_tensor[:, :, 12].contiguous()

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
    ), 2)
    var_embed = var_embed.view(batch_size, -1, window_size, fea_len)
    if opt.use_gpu:
        label_tensor = torch.from_numpy(conv_to_ndarray([tag_to_idx(e) for e in labels])).cuda()
    else:
        label_tensor = torch.from_numpy(conv_to_ndarray([tag_to_idx(e) for e in labels]))
    return (var_embed, label_tensor)

def train_svm(model, X_train, y_train, rootPath = '/home/dsk/traffic_model/checkpoints'):
    """if os.path.exists(os.path.join(rootPath,'svm_param_1.m')):
        model = joblib.load(os.path.join(rootPath,'svm_param_1.m'))
        return model
    """
    #Data Preprocessing
    X,y = conv_to_ndarray(X_train), conv_to_ndarray(y_train)
    #calculate differences of timeintervals
    for i in range(len(X)):
        start_time = X[i][0, -1]
        for j in range(len(X[i])):
            prev = X[i][j,-1]
            X[i][j,-1] -= start_time
            start_time = prev
    #generate feature vector
    new_X = [genStatisVec2(x, True, True) for x in X]
    #normalization
    scaler = StandardScaler()
    scaler.fit(new_X)
    new_X = scaler.transform(new_X)
    #training
    model.fit(new_X, y)
    #get training result
    score = model.score(new_X, y)
    print 'SVM accuracy on training data is', score

    joblib.dump(model, os.path.join(rootPath,'svm_param.m'))
    print 'svm params saved successfully.'

    return model

def val_svm(model, X_test, y_test, mix=1, level=1):
    #Data Preprocessing
    assert mix>0
    X,y = conv_to_ndarray(X_test), conv_to_ndarray(y_test)
    if mix==1:
        #calculate differences
        for i in range(len(X)):
            start_time = X[i][0, -1]
            for j in range(len(X[i])):
                prev = X[i][j, -1]
                X[i][j, -1] -= start_time
                start_time = prev
        # generate feature vector
        new_X = [genStatisVec2(x, True, True) for x in X]
        # normalization
        scaler = StandardScaler()
        scaler.fit(new_X)
        new_X = scaler.transform(new_X)
        #evaluate the results
        print 'SVM accuracy on testing data (mix = 1 )', model.score(new_X, y)

    else:
        # generate validation set
        if level==1:
            validate_set = gen_mixture(X, y, len(X), mix, algorithm='svm', ip=True, port=True)
        else:
            validate_set = gen_hardmix(X, y, len(X), mix, algorithm='svm', ip=True, port=True)
        fs,ls = zip(*validate_set)
        #return value will be np.array, so do 'tolist()'
        fs,ls = list(fs), list(ls)
        for i in range(len(ls)):
            if type(ls[i]) == np.float64:
                ls[i] = [ls[i]]
        # do normalization
        scaler = StandardScaler()
        for i in range(len(fs)):
            scaler.fit(fs[i])
            fs[i] = scaler.transform(fs[i])
        # validation
        correct, false_p, false_n, total = 0, 0, 0, 0
        for i in range(len(fs)):
            res = []
            for j in range(len(fs[i])):
                tmp_res = zip(model.decision_function([fs[i][j]])[0].tolist(), range(30))
                """
                The type of svm.decision_function([x]) is numpy.ndarray-->array([[x x x x]]), therefore if you zip(decision_function([x]), range(len(x)))
                you will get [(array([x x x x]),0)]
                """
                tmp_res.sort(key=lambda x: x[0], reverse=True)
                #tmp_res = conv_to_ndarray(tmp_res[:mix])[:, 1]
                tmp_res = tmp_res[0][1]
                #res.extend(tmp_res)
                res.append(tmp_res)
            m = set(res) & set(ls[i])
            p = set(res) - set(ls[i])
            n = set(ls[i]) - set(res)

            correct += len(m)
            false_p += len(p)
            false_n += len(n)

            total += len(ls[i])
        print 'SVM evalutation on testing data mix =',mix,' acc ',1.0*correct/total, \
            ' false P', 1.0*false_p/total, 'false N', 1.0*false_n/total

def wrapper_svm(model, X_train, y_train, X_test, y_test):
    train_svm(model, X_train, y_train)
    val_svm(model, X_test, y_test, 1)
    val_svm(model, X_test, y_test, 2, 2)
    val_svm(model, X_test, y_test, 4, 2)

def train_CNN(model, X_train, y_train, load_path = '/home/dsk/traffic_model/checkpoints'):
    opt = DefaultConfig()
    # use torch.utils.DataLoader to speedup the data processing
    trContainer = trafficContainer(X_train, y_train)
    trDataloader = DataLoader(trContainer, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)

    if not os.path.exists(os.path.join(load_path, model.name)):
        if opt.use_gpu:
            model.cuda()
            loss_function = nn.CrossEntropyLoss().cuda()
        else:
            loss_function = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        for epoch in range(opt.max_epoch):
            total, correct = 0, 0
            print 'epoch', epoch
            for i, batch in enumerate(trDataloader):
                if opt.use_gpu:
                    sample_tensor =batch['window'].cuda() #batch['window'].type(torch.cuda.FloatTensor)
                else:
                    sample_tensor = batch['window']
                labels = batch['label']
                # Here calculate differences of timeintervals
                for k in range(len(sample_tensor)):
                    start_time = sample_tensor[k][0, -1]
                    for j in range(len(sample_tensor[k])):
                        prev = sample_tensor[k][j, -1]
                        sample_tensor[k][j, -1] -= start_time
                        start_time = prev
                var_embed, label_tensor = trans_embed(model, sample_tensor, labels)
                #print 'var_embed.size()', var_embed.size()
                model.zero_grad()
                last_output = model(var_embed)

                _, predicted = torch.max(last_output.data, 1)
                total += label_tensor.size(0)
                correct += (predicted == label_tensor).sum()

                loss = loss_function(last_output, autograd.Variable(label_tensor))
                loss.backward(retain_graph=True)
                optimizer.step()

            print 'epoch ', epoch, 'accuracy ', 1.0*correct/total
        # whenever save model state dict, use model.cpu() to have a try.
        model.cpu()
        torch.save(model.state_dict(), os.path.join(load_path, model.name))
        print 'model saved successfully.'
    else:
        model.load_state_dict(torch.load(os.path.join(load_path, model.name)))

    if opt.use_gpu:
        model.cuda()
    #Do simple validation on "training set"
    correct, total = 0, 0
    for i, batch in enumerate(trDataloader):
        if opt.use_gpu:
            sample_tensor = batch['window'].cuda()  # batch['window'].type(torch.cuda.FloatTensor)
        else:
            sample_tensor = batch['window']
        labels = batch['label']
    # Here calculate differences of timeintervals
        for k in range(len(sample_tensor)):
            start_time = sample_tensor[k][0, -1]
            for j in range(len(sample_tensor[k])):
                prev = sample_tensor[k][j, -1]
                sample_tensor[k][j, -1] -= start_time
                start_time = prev
        var_embed, label_tensor = trans_embed(model, sample_tensor, labels)
        model.eval()
        result = model(var_embed)
        _, predicted = torch.max(result.data, 1)
        total += label_tensor.size(0)
        correct += (predicted == label_tensor).sum()
    print 'CNN accuracy on training data is', 1.0*correct/total

def val_CNN(model, X_test, y_test, mix=1, level=1):
    assert mix>0
    opt = DefaultConfig()
    correct, false_p, false_n, total = 0, 0, 0, 0
    model.eval()
    if opt.use_gpu:
        model.cuda()
    if mix==1:
        teContainer = trafficContainer(X_test, y_test)
        teDataloader = DataLoader(teContainer, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)
        for i, batch in enumerate(teDataloader):
            if opt.use_gpu:
                sample_tensor =batch['window'].cuda() #batch['window'].type(torch.cuda.FloatTensor)
            else:
                sample_tensor = batch['window']
            labels = batch['label']
            # Here calculate differences of timeintervals
            for k in range(len(sample_tensor)):
                start_time = sample_tensor[k][0, -1]
                for j in range(len(sample_tensor[k])):
                    prev = sample_tensor[k][j, -1]
                    sample_tensor[k][j, -1] -= start_time
                    start_time = prev
            var_embed, label_tensor = trans_embed(model, sample_tensor, labels)
            result = model(var_embed)
            _, predicted = torch.max(result.data, 1)
            correct += (predicted == label_tensor).sum()
            total += label_tensor.size(0)
        print 'CNN accuracy on testing data is ', 1.0*correct/total
    else:
        X, y = conv_to_ndarray(X_test), conv_to_ndarray(y_test)
        if level==1:
            validate_set = gen_mixture(X, y, len(X), mix, ip=True, port=True)
        else:
            validate_set = gen_hardmix(X, y, len(X), mix, ip=True, port=True)
        fs, ls = zip(*validate_set)
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
            sample_tensor = sample_tensor.view(1, sample_tensor.size(0), -1)
            batch_size, window_size, feature_len = sample_tensor.size()

            if opt.use_gpu:
                sample_tensor = sample_tensor.cuda()
            sample_tensor, label_tensor = trans_embed(model, sample_tensor, label_tensor)

            #divide sample_tensor into slicings.
            ws = window_size/mix
            slicings = [
                sample_tensor[:,:,int(i*ws):int(i*ws+ws),:] for i in range(mix)
            ]
            results = []
            for slice in slicings:
                scores = model(slice)
                largest = torch.max(scores.data, 1)[1]
                results.extend(list(largest))
            #print 'results', results
            m = set(results)&set(label_tensor)
            p = set(results)-set(label_tensor)
            n = set(label_tensor)-set(results)
            correct += len(m)
            false_p += len(p)
            false_n += len(n)
            total += len(label_tensor)
        print 'CNN evalutation on testing data mix =',mix,' acc ',1.0*correct/total, \
            ' false P', 1.0*false_p/total, 'false N', 1.0*false_n/total

def wrapper_CNN(model, X_train, y_train, X_test, y_test):
    train_CNN(model, X_train, y_train)
    val_CNN(model, X_test, y_test, 1)
    val_CNN(model, X_test, y_test, 2, 2)
    val_CNN(model, X_test, y_test, 4, 2)

def train_LSTM(model, X_train, y_train, load_path = '/home/dsk/traffic_model/checkpoints'):
    opt = DefaultConfig()
    # use torch.utils.DataLoader to speedup the data processing
    trContainer = trafficContainer(X_train, y_train)
    trDataloader = DataLoader(trContainer, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)
    if opt.use_gpu:
        model.cuda()
        loss_function = nn.CrossEntropyLoss().cuda()
    else:
        loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if not os.path.exists(os.path.join(load_path, model.name)):
        for epoch in range(opt.max_epoch):
            correct, total = 0, 0
            print 'epoch', epoch
            for i, batch in enumerate(trDataloader):
                if opt.use_gpu:
                    sample_tensor =batch['window'].cuda() #batch['window'].type(torch.cuda.FloatTensor)
                else:
                    sample_tensor = batch['window']
                labels = batch['label']
                # Here calculate differences of timeintervals
                for k in range(len(sample_tensor)):
                    start_time = sample_tensor[k][0, -1]
                    for j in range(len(sample_tensor[k])):
                        prev = sample_tensor[k][j, -1]
                        sample_tensor[k][j, -1] -= start_time
                        start_time = prev
                var_embed, label_tensor = trans_embed(model, sample_tensor, labels)
                # you need to squeeze var_embed's second dimension
                var_embed = torch.squeeze(var_embed, 1)
                model.zero_grad()
                last_output = model(var_embed)

                # Here calculate the accuracy at each epoch
                _, predicted = torch.max(last_output.data, 1)
                total += label_tensor.size(0)
                correct += (predicted == label_tensor).sum()

                loss = loss_function(last_output, autograd.Variable(label_tensor))
                loss.backward(retain_graph=True)
                optimizer.step()
            print 'epoch ', epoch, 'accuracy ', 1.0*correct/total
        model.cpu()
        torch.save(model.state_dict(), os.path.join(load_path, model.name))
        print 'model saved successfully.'
    else:
        model.load_state_dict(torch.load(os.path.join(load_path, model.name)))
    #Do simple validation on "training set"
    if opt.use_gpu:
        model.cuda()

    correct, total = 0, 0
    for i, batch in enumerate(trDataloader):
        if opt.use_gpu:
            sample_tensor = batch['window'].cuda()  # batch['window'].type(torch.cuda.FloatTensor)
        else:
            sample_tensor = batch['window']
        labels = batch['label']
    # Here calculate differences of timeintervals
        for k in range(len(sample_tensor)):
            start_time = sample_tensor[k][0, -1]
            for j in range(len(sample_tensor[k])):
                prev = sample_tensor[k][j, -1]
                sample_tensor[k][j, -1] -= start_time
                start_time = prev
        var_embed, label_tensor = trans_embed(model, sample_tensor, labels)
        # squeeze var_embed's second dimension
        var_embed = torch.squeeze(var_embed, 1)
        model.eval()
        result = model(var_embed)
        _, predicted = torch.max(result.data, 1)
        total += label_tensor.size(0)
        correct += (predicted == label_tensor).sum()
    print 'LSTM accuracy on training data is', 1.0*correct/total

def val_LSTM(model, X_test, y_test, mix=1, level=1):
    assert mix>0
    opt = DefaultConfig()
    correct, false_p, false_n, total = 0, 0, 0, 0
    model.eval()
    if opt.use_gpu:
        model.cuda()
    if mix==1:
        teContainer = trafficContainer(X_test, y_test)
        teDataloader = DataLoader(teContainer, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)
        for i, batch in enumerate(teDataloader):
            if opt.use_gpu:
                sample_tensor =batch['window'].cuda() #batch['window'].type(torch.cuda.FloatTensor)
            else:
                sample_tensor = batch['window']
            labels = batch['label']
            # Here calculate differences of timeintervals
            for k in range(len(sample_tensor)):
                start_time = sample_tensor[k][0, -1]
                for j in range(len(sample_tensor[k])):
                    prev = sample_tensor[k][j, -1]
                    sample_tensor[k][j, -1] -= start_time
                    start_time = prev
            var_embed, label_tensor = trans_embed(model, sample_tensor, labels)
            #squeeze the second dimension
            var_embed = torch.squeeze(var_embed, 1)

            result = model(var_embed)
            _, predicted = torch.max(result.data, 1)
            correct += (predicted == label_tensor).sum()
            total += label_tensor.size(0)
        print 'LSTM accuracy on testing data is ', 1.0*correct/total
    else:
        X, y = conv_to_ndarray(X_test), conv_to_ndarray(y_test)
        if level==1:
            validate_set = gen_mixture(X, y, len(X), mix, ip=True, port=True)
        else:
            validate_set = gen_hardmix(X, y, len(X), mix, ip=True, port=True)
        fs, ls = zip(*validate_set)
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
            sample_tensor = sample_tensor.view(1, sample_tensor.size(0), -1)
            batch_size, window_size, feature_len = sample_tensor.size()

            if opt.use_gpu:
                sample_tensor = sample_tensor.cuda()
            var_embed, label_tensor = trans_embed(model, sample_tensor, label_tensor)
            var_embed = torch.squeeze(var_embed, 1)

            ws = window_size/mix
            slicings = [
                var_embed[:,int(i*ws):int(i*ws+ws),:] for i in range(mix)
            ]
            results = []
            for slice in slicings:
                scores = model(slice)
                largest = torch.max(scores.data, 1)[1]
                results.extend(list(largest))
            #scores = model(var_embed)
            #results = scores.sort(descending=True)[-1].view(-1)[:mix].data
            #print 'results', results
            m = set(results)&set(label_tensor)
            p = set(results)-set(label_tensor)
            n = set(label_tensor)-set(results)
            correct += len(m)
            false_p += len(p)
            false_n += len(n)
            total += len(label_tensor)
        print 'LSTM evalutation on testing data mix =',mix,' acc ',1.0*correct/total, \
            ' false P', 1.0*false_p/total, 'false N', 1.0*false_n/total

def wrapper_LSTM(model, X_train, y_train, X_test, y_test):
    train_LSTM(model, X_train, y_train)
    val_LSTM(model, X_test, y_test, 1)
    val_LSTM(model, X_test, y_test, 2, 2)
    val_LSTM(model, X_test, y_test, 4, 2)

def train_CNN_LSTM(m_cnn, m_lstm, X_train, y_train):
    opt = DefaultConfig()
    m_cnn.train()
    m_lstm.train()
    # use torch.utils.DataLoader to speedup the data processing
    trContainer = trafficContainer(X_train, y_train)
    trDataloader = DataLoader(trContainer, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)
    if opt.use_gpu:
        m_cnn.cuda()
        m_lstm.cuda()
        loss_function = nn.CrossEntropyLoss().cuda()
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(m_cnn.parameters(), lr=opt.lr)
    for epoch in range(opt.max_epoch):
        correct, total = 0, 0
        print 'epoch', epoch
        for i, batch in enumerate(trDataloader):
            if opt.use_gpu:
                sample_tensor =batch['window'].cuda() #batch['window'].type(torch.cuda.FloatTensor)
            else:
                sample_tensor = batch['window']
            labels = batch['label']
            # Here calculate differences of timeintervals
            for k in range(len(sample_tensor)):
                start_time = sample_tensor[k][0, -1]
                for j in range(len(sample_tensor[k])):
                    prev = sample_tensor[k][j, -1]
                    sample_tensor[k][j, -1] -= start_time
                    start_time = prev
            var_embed, label_tensor = trans_embed(m_cnn, sample_tensor, labels)
            batch_size, window_size, feature_len = var_embed.size(0), var_embed.size(2), var_embed.size(3)
            #var_embed = var_embed.view(-1, 1, m_cnn.window_size, feature_len)
            vars, start = [], 0
            while(start+m_cnn.window_size<window_size):
                vars.append(var_embed[:,:,start:start+m_cnn.window_size,:])
                start += 2
            in_size = var_embed.size(0)
            """original---> hiddens = m_cnn.fc2hidden(
                m_cnn.dropout(m_cnn.layer3(m_cnn.layer2(m_cnn.layer1(
                    var_embed
                )))).view(in_size, -1)
            )"""
            hiddens = [
                m_cnn.hidden2tag(
                    m_cnn.fc2hidden(
                        m_cnn.layer3(m_cnn.layer2(m_cnn.layer1(
                            var
                        ))).view(in_size, -1)
                    )
                )for var in vars
            ]
            hiddens_tensor = torch.stack(hiddens)
            #hiddens_tensor.size() --> (num_windows, batch_size, feature_len)
            #we need to transpose 0 and 1 dimensions
            hiddens_tensor = hiddens_tensor.transpose(1, 0)
            """original--> hidden_len = hiddens.size(1)
            hiddens = hiddens.view(batch_size, -1, hidden_len)"""
            last_output = m_lstm(hiddens_tensor)

            # Here calcuate the accuracy at each epoch.
            _, predicted = torch.max(last_output.data, 1)
            total += label_tensor.size(0)
            correct += (predicted == label_tensor).sum()

            loss = loss_function(last_output, autograd.Variable(label_tensor))
            loss.backward(retain_graph=True)
            optimizer.step()
        print 'epoch ', epoch, 'accuracy ', 1.0*correct/total

    # Do validation on training set
    m_cnn.eval()
    m_lstm.eval()
    correct, total = 0, 0
    for i, batch in enumerate(trDataloader):
        if opt.use_gpu:
            sample_tensor = batch['window'].cuda()  # batch['window'].type(torch.cuda.FloatTensor)
        else:
            sample_tensor = batch['window']
        labels = batch['label']
        # Here calculate differences of timeintervals
        for k in range(len(sample_tensor)):
            start_time = sample_tensor[k][0, -1]
            for j in range(len(sample_tensor[k])):
                prev = sample_tensor[k][j, -1]
                sample_tensor[k][j, -1] -= start_time
                start_time = prev
        var_embed, label_tensor = trans_embed(m_cnn, sample_tensor, labels)
        batch_size, window_size, feature_len = var_embed.size(0), var_embed.size(2), var_embed.size(3)
        vars, start = [], 0
        while (start + m_cnn.window_size < window_size):
            vars.append(var_embed[:, :, start:start + m_cnn.window_size, :])
            start += 2
        in_size = var_embed.size(0)
        """original---> hiddens = m_cnn.fc2hidden(
            m_cnn.dropout(m_cnn.layer3(m_cnn.layer2(m_cnn.layer1(
                var_embed
            )))).view(in_size, -1)
        )"""
        hiddens = [
            m_cnn.fc2hidden(
                m_cnn.layer3(m_cnn.layer2(m_cnn.layer1(
                    var
                ))).view(in_size, -1)
            ) for var in vars
            ]
        hiddens_tensor = torch.stack(hiddens)
        hiddens_tensor = hiddens_tensor.transpose(1, 0)
        result = m_lstm(hiddens_tensor)
        _, predicted = torch.max(result.data, 1)
        total += label_tensor.size(0)
        correct += (predicted == label_tensor).sum()
    print 'CNN-LSTM accuracy on training data is', 1.0*correct/total

def val_CNN_LSTM(m_cnn, m_lstm, X_test, y_test, mix=1, level=1):
    assert mix>0
    opt = DefaultConfig()
    correct, false_p, false_n, total = 0, 0, 0, 0
    m_cnn.eval()
    m_lstm.eval()
    if opt.use_gpu:
        m_cnn.cuda()
        m_lstm.cuda()
    if mix == 1:
        teContainer = trafficContainer(X_test, y_test)
        teDataloader = DataLoader(teContainer, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)
        for i, batch in enumerate(teDataloader):
            if opt.use_gpu:
                sample_tensor = batch['window'].cuda()  # batch['window'].type(torch.cuda.FloatTensor)
            else:
                sample_tensor = batch['window']
            labels = batch['label']
            # Here calculate differences of timeintervals
            for k in range(len(sample_tensor)):
                start_time = sample_tensor[k][0, -1]
                for j in range(len(sample_tensor[k])):
                    prev = sample_tensor[k][j, -1]
                    sample_tensor[k][j, -1] -= start_time
                    start_time = prev
            var_embed, label_tensor = trans_embed(m_cnn, sample_tensor, labels)
            batch_size, window_size, feature_len = var_embed.size(0), var_embed.size(2), var_embed.size(3)
            vars, start = [], 0
            while(start+m_cnn.window_size<window_size):
                vars.append(var_embed[:,:,start:start+m_cnn.window_size,:])
                start += 1
            in_size = var_embed.size(0)
            hiddens = [
                m_cnn.fc2hidden(
                    m_cnn.layer3(m_cnn.layer2(m_cnn.layer1(
                        var
                    ))).view(in_size, -1)
                ) for var in vars
            ]
            hiddens_tensor = torch.stack(hiddens)
            #hiddens_tensor.size() --> (num_windows, batch_size, feature_len)
            #we need to transpose 0 and 1 dimensions
            hiddens_tensor = hiddens_tensor.transpose(1, 0)
            result = m_lstm(hiddens_tensor)
            _, predicted = torch.max(result.data, 1)
            total += label_tensor.size(0)
            correct += (predicted == label_tensor).sum()
        print 'CNN-LSTM accuracy on testing data is ', 1.0 * correct / total
    else:
        X, y = conv_to_ndarray(X_test), conv_to_ndarray(y_test)
        if level==1:
            validate_set = gen_mixture(X, y, len(X), mix, ip=True, port=True)
        else:
            validate_set = gen_hardmix(X, y, len(X), mix, ip=True, port=True)
        fs, ls = zip(*validate_set)
        for i in range(len(fs)):
            sample_tensor = fs[i]
            labels = ls[i]
            if type(labels) == np.float64:
                labels = [labels]
            labels = [tag_to_idx(e) for e in labels]

            sample_tensor = conv_to_ndarray(sample_tensor)
            sample_tensor = torch.from_numpy(sample_tensor)
            sample_tensor = sample_tensor.type(torch.FloatTensor)
            sample_tensor = sample_tensor.view(1, sample_tensor.size(0), -1)
            var_embed, label_tensor = trans_embed(m_cnn, sample_tensor, labels)
            batch_size, window_size, feature_len = var_embed.size(0), var_embed.size(2), var_embed.size(3)
            vars, start = [], 0
            while(start+m_cnn.window_size<window_size):
                vars.append(var_embed[:,:,start:start+m_cnn.window_size,:])
                start += 1
            in_size = var_embed.size(0)
            hiddens = [
                m_cnn.fc2hidden(
                    m_cnn.layer3(m_cnn.layer2(m_cnn.layer1(
                        var
                    ))).view(in_size, -1)
                ) for var in vars
            ]
            hiddens_tensor = torch.stack(hiddens)
            #hiddens_tensor.size() --> (num_windows, batch_size, feature_len)
            #we need to transpose 0 and 1 dimensions
            hiddens_tensor = hiddens_tensor.transpose(1, 0)
            scores = m_lstm(hiddens_tensor)
            results = scores.sort(descending=True)[-1].view(-1)[:mix].data
            #print 'results', results
            m = set(results)&set(label_tensor)
            p = set(results)-set(label_tensor)
            n = set(label_tensor)-set(results)
            correct += len(m)
            false_p += len(p)
            false_n += len(n)
            total += len(label_tensor)
        print 'CNN-LSTM evalutation on testing data mix =', mix, ' acc ', 1.0 * correct / total, \
            ' false P', 1.0 * false_p / total, 'false N', 1.0 * false_n / total
if __name__ == "__main__":
    """
    readers = [
    trafficReader(i, True) for i in range(30)
    ]
    X_train, X_test, y_train, y_test = genDataset(readers, 10000, 20, 10)
    print len(X_train),len(X_test),len(y_train),len(y_test)

    model_cnn2 = CNN2(20, 71, 128, 30)
    train_CNN(model_cnn2, X_train, y_train)
    val_CNN(model_cnn2, X_test, y_test)

    X_train2, X_test2, y_train2, y_test2 = genDataset(readers, 4000, 100, 20)
    print len(X_train), len(X_test), len(y_train), len(y_test)

    model_lstm = LSTM(30, 16, 30, 2)
    train_CNN_LSTM(model_cnn2, model_lstm, X_train2, y_train2)
    val_CNN_LSTM(model_cnn2, model_lstm, X_test2, y_test2)
    val_CNN_LSTM(model_cnn2, model_lstm, X_test2, y_test2, 4, 2)

    # linearLSTM and CNN_LSTM
    print 'torch.__version__', torch.__version__
    readers = [
        trafficReader(i, True) for i in range(30)
    ]
    X_train, X_test, y_train, y_test = genDataset(readers, 5000, 500, 20)
    print len(X_train),len(X_test),len(y_train),len(y_test)
    model_liLSTM = LinearLSTM(71, 128, 16, 30, 2)
    wrapper_LSTM(model_liLSTM, X_train, y_train, X_test, y_test)
    model_cnnlstm = CNN_LSTM(20, 71, 64, 128, 16, 30, 2)
    wrapper_LSTM(model_cnnlstm, X_train, y_train, X_test, y_test)
    """

    """
    # CNN_LSTM2
    readers = [
    trafficReader(i, True) for i in range(30)
    ]
    X_train, X_test, y_train, y_test = genDataset(readers, 4000, 100, 20)
    print len(X_train),len(X_test),len(y_train),len(y_test)
    model_cnnlstm = CNN_LSTM2(71, 64, 128, 16, 30, 2)
    wrapper_CNN(model_cnnlstm, X_train, y_train, X_test, y_test)
    """



    # SVM, CNN, LSTM and CNNLSTM
    readers = [
        trafficReader(i, True) for i in range(30)
    ]
    X_train, X_test, y_train, y_test = genDataset(readers, 4000, 100, 30)
    model_svm = svm.SVC(decision_function_shape='ovr')
    #model_cnn = CNN(71, 30, 30)
    model_lstm = LSTM(71, 16, 32, 30, 2, 30)
    #model_liLSTM = LinearLSTM(71, 128, 16, 32, 30, 2, 30)
    #model_cnnlstm = CNN_LSTM2(71, 64, 128, 16, 30, 2, 30)
    p1 = Process(target=wrapper_svm, args=(model_svm, X_train, y_train, X_test, y_test))
    p1.start()
    #wrapper_svm(model_svm, X_train, y_train, X_test, y_test)
    #wrapper_CNN(model_cnn, X_train, y_train, X_test, y_test)
    wrapper_LSTM(model_lstm, X_train, y_train, X_test, y_test)
    #wrapper_LSTM(model_liLSTM, X_train, y_train, X_test, y_test)
    #wrapper_CNN(model_cnnlstm, X_train, y_train, X_test, y_test)
    p1.join()

    """
    # CNN and LSTM
    readers = [
        trafficReader(i, True) for i in range(30)
    ]
    X_train, X_test, y_train, y_test = genDataset(readers, 30, 100, 20)
    model_lstm = LSTM(71, 16, 30, 2, True)
    model_cnn = CNN(71, 30)
    wrapper_LSTM(model_lstm, X_train, y_train, X_test, y_test)
    wrapper_CNN(model_cnn, X_train, y_train, X_test, y_test)
    """








