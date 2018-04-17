import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from ..config import DefaultConfig
import numpy as np

from ..utils import read_params
from BasicModule import BasicModule

class LSTM(nn.Module):
    """
    In this model, I used newly-generated data in /home/4tshare/iot/infocomm_data/category_v3
    It has ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','datalen','timeval'] features.
    Due to tcp has a hand-shaking procedure, I checked the PSH flag in tcp header to get the tcp packets really transferring data instead of those establishing connections.

    Also some other data cleaning work should be done to remove those packets only existing in LAN environments. like ARP packets.
    """
    def __init__(self, feature_dim, hidden_dim, linear_dim, tagset_size, num_layers, embed_dim, cuda_support=True):
        super(LSTM, self).__init__()
        self.name = "lstm_basic"
        self.opt = DefaultConfig()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        ipdict, portdict, = read_params()
        self.ipembed = nn.Embedding(len(ipdict)+1,embed_dim)
        self.portembed = nn.Embedding(len(portdict)+1,embed_dim)
        #self.protembed = nn.Embedding(len(protdict)+1,10)
        """
        nn.Embedding can only handle autograd.Variable
        """
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5, bidirectional=False)
        #self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        self.hidden2linear = nn.Linear(hidden_dim, linear_dim)
        self.linear2tag = nn.Linear(linear_dim, tagset_size)
        """
        LSTM:
        inputs: input, (h_0, c_0)
        > input (seq_len, batch, input_size)
        > h_0 (num_layers*num_directions, batch, hidden_size)
        > c_0 (num_layers*num_directions, batch, hidden_size)
        outputs: output, (h_n, c_n)
        > output (seq_len, batch, hidden_size*num_directions)
        """

    def forward(self, var_embed):
        """

        :param var_embed: (batch_size, window_size, feature_len) -> torch.(cuda).FloatTensor
        :return:
        """
        batch_size, window_size, feature_len = var_embed.size()
        #h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers*2, batch_size, self.hidden_dim)).cuda())
        #c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers*2, batch_size, self.hidden_dim)).cuda())
        if self.opt.use_gpu:
            h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers, batch_size, self.hidden_dim)).cuda())
            c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers, batch_size, self.hidden_dim)).cuda())
        else:
            h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers, batch_size, self.hidden_dim)))
            c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers, batch_size, self.hidden_dim)))

        #print 'vec_seq', vec_seq (batch_size, window_size, feature_len)
        #print 'vec_seq[:,:,0].type(torch.cuda.LongTensor)', vec_seq[:,:,0].type(torch.cuda.LongTensor)
        #print 'ipe', ipe#, ipe (batch_size, window_size, 10)
        if self.opt.use_gpu:
            var_embed.cuda()
        #print 'var_embed', var_embed (batch_size, window_size, 33)
        #lstm_out, self.hidden = self.lstm(var_embed, self.hidden)
        #UserWarning: RNN module weights are not part of single contiguous chunk of memory.
        #self.lstm.flatten_parameters()
        lstm_output, (ht, ct) = self.lstm(var_embed, (h0, c0))
        #print 'lstm_out.size() = ', lstm_out.size() # (batch_size,window_size,hidden_dim*2)

        tag_space = self.linear2tag(self.hidden2linear(ht[-1]))
        tag_scores = F.log_softmax(tag_space)
        #print 'tag_scores', tag_scores
        return tag_scores

class LinearLSTM(nn.Module):
    """
    In this model, I used newly-generated data in /home/4tshare/iot/infocomm_data/category_v3
    It has ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','datalen','timeval'] features.
    Due to tcp has a hand-shaking procedure, I checked the PSH flag in tcp header to get the tcp packets really transferring data instead of those establishing connections.

    Also some other data cleaning work should be done to remove those packets only existing in LAN environments. like ARP packets.
    """
    def __init__(self, feature_dim, lstm_input_dim, hidden_dim, linear_dim, tagset_size, num_layers, embed_dim, cuda_support=True):
        super(LinearLSTM, self).__init__()
        self.name = "linearLSTM"
        self.opt = DefaultConfig()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        ipdict, portdict, = read_params()
        self.ipembed = nn.Embedding(len(ipdict)+1,embed_dim)
        self.portembed = nn.Embedding(len(portdict)+1,embed_dim)
        #self.protembed = nn.Embedding(len(protdict)+1,10)
        """
        nn.Embedding can only handle autograd.Variable
        """
        self.linear = nn.Linear(feature_dim, lstm_input_dim)
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5, bidirectional=False)
        #self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        self.hidden2linear = nn.Linear(hidden_dim, linear_dim)
        self.linear2tag = nn.Linear(linear_dim, tagset_size)
        """
        LSTM:
        inputs: input, (h_0, c_0)
        > input (seq_len, batch, input_size)
        > h_0 (num_layers*num_directions, batch, hidden_size)
        > c_0 (num_layers*num_directions, batch, hidden_size)
        outputs: output, (h_n, c_n)
        > output (seq_len, batch, hidden_size*num_directions)
        """

    def forward(self, var_embed):
        """

        :param var_embed: (batch_size, window_size, feature_len) -> torch.(cuda).FloatTensor
        :return:
        """
        batch_size, window_size, feature_len = var_embed.size()
        #h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers*2, batch_size, self.hidden_dim)).cuda())
        #c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers*2, batch_size, self.hidden_dim)).cuda())
        if self.opt.use_gpu:
            h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers, batch_size, self.hidden_dim)).cuda())
            c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers, batch_size, self.hidden_dim)).cuda())
        else:
            h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers, batch_size, self.hidden_dim)))
            c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers, batch_size, self.hidden_dim)))

        #print 'vec_seq', vec_seq (batch_size, window_size, feature_len)
        #print 'vec_seq[:,:,0].type(torch.cuda.LongTensor)', vec_seq[:,:,0].type(torch.cuda.LongTensor)
        #print 'ipe', ipe#, ipe (batch_size, window_size, 10)
        if self.opt.use_gpu:
            var_embed.cuda()
        #print 'var_embed', var_embed (batch_size, window_size, 33)
        #lstm_out, self.hidden = self.lstm(var_embed, self.hidden)
        #UserWarning: RNN module weights are not part of single contiguous chunk of memory.
        ipt = self.linear(var_embed)
        self.lstm.flatten_parameters()
        lstm_output, (ht, ct) = self.lstm(ipt, (h0, c0))
        #print 'lstm_out.size() = ', lstm_out.size() # (batch_size,window_size,hidden_dim*2)

        tag_space = self.linear2tag(self.hidden2linear(ht[-1]))
        tag_scores = F.log_softmax(tag_space)
        #print 'tag_scores', tag_scores
        return tag_scores

class CNN_LSTM(nn.Module):
    def __init__(self, cnn_window_size, feature_dim, cnn_hidden_dim, lstm_input_dim, lstm_hidden_dim, tagset_size, lstm_num_layers, embed_dim):
        super(CNN_LSTM, self).__init__()
        self.opt = DefaultConfig()
        ipdict, portdict = read_params()
        self.cnn_window_size = cnn_window_size
        self.feature_dim = feature_dim
        self.cnn_hidden_dim = cnn_hidden_dim
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.tagset_size = tagset_size
        self.lstm_num_layers = lstm_num_layers

        self.ipembed = nn.Embedding(len(ipdict)+1,embed_dim)
        self.portembed = nn.Embedding(len(portdict)+1,embed_dim)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc2hidden = nn.Linear(448, cnn_hidden_dim)
        self.hidden2lstm = nn.Linear(cnn_hidden_dim, lstm_input_dim)

        self.lstm =  nn.LSTM(lstm_input_dim, lstm_hidden_dim, lstm_num_layers, batch_first=True, dropout=0.5, bidirectional=False)
        self.lstm2tag = nn.Linear(lstm_hidden_dim, tagset_size)

    def forward(self, var_embed):
        #print 'var_embed.size()', var_embed.size()
        var_embed = torch.unsqueeze(var_embed, 1)
        batch_size, window_size, feature_len = var_embed.size(0), var_embed.size(2), var_embed.size(3)
        if self.opt.use_gpu:
            h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_layers, batch_size, self.lstm_hidden_dim)).cuda())
            c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_layers, batch_size, self.lstm_hidden_dim)).cuda())
        else:
            h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_layers, batch_size, self.lstm_hidden_dim)))
            c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_layers, batch_size, self.lstm_hidden_dim)))
        vars, start = [], 0
        while (start + self.cnn_window_size < window_size):
            vars.append(var_embed[:, :, start:start + self.cnn_window_size, :])
            start += 10
        in_size = var_embed.size(0)
        hs = [
            self.hidden2lstm(
                self.fc2hidden(
                    self.dropout(
                        self.layer3(self.layer2(self.layer1(
                            var
                        ))).view(in_size, -1)
                    )
                )
            ) for var in vars
        ]
        h_tensor = torch.stack(hs)
        h_tensor = h_tensor.transpose(1,0)
        self.lstm.flatten_parameters()
        lstm_output, (ht, ct) = self.lstm(h_tensor, (h0, c0))
        #print 'ht.size()', ht.size()
        tag_space = self.lstm2tag(ht[-1])
        #print 'tag_space.size()', tag_space.size()
        tag_scores = F.log_softmax(tag_space)
        #print 'tag_scores.size()', tag_scores.size()

        return tag_scores


class CNN_LSTM2(nn.Module):
    def __init__(self, feature_dim, cnn_hidden_dim, lstm_input_dim, lstm_hidden_dim, tagset_size, lstm_num_layers, embed_dim):
        super(CNN_LSTM2, self).__init__()
        self.opt = DefaultConfig()
        self.name = 'cnn_lstm2'
        ipdict, portdict = read_params()
        self.feature_dim = feature_dim
        self.cnn_hidden_dim = cnn_hidden_dim
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.tagset_size = tagset_size
        self.lstm_num_layers = lstm_num_layers

        self.c1_num = 16
        self.c2_num = 32
        self.c3_num = 64

        self.ipembed = nn.Embedding(len(ipdict)+1,embed_dim)
        self.portembed = nn.Embedding(len(portdict)+1,embed_dim)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(0.5)
        if self.opt.use_gpu:
            self.lstms = [nn.LSTM(7, 16, 2, batch_first=True, dropout=0.5).cuda() for i in range(self.c3_num)]
        else:
            self.lstms = [nn.LSTM(7, 16, 2, batch_first=True, dropout=0.5)for i in range(self.c3_num)]
        print 'has already define lstms'
        self.lstm2tag = nn.Linear(64*16, tagset_size)

    def forward(self, var_embed):
        #var_embed = torch.unsqueeze(var_embed, 1)
        batch_size, window_size, feature_len = var_embed.size(0), var_embed.size(2), var_embed.size(3)
        out = self.dropout(self.layer3(
            self.layer2(
                self.layer1(var_embed)
            )
        ))
        lstm_input = out
        lstm_outputs = []
        if self.opt.use_gpu:
            h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_layers, batch_size, self.lstm_hidden_dim)).cuda())
            c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_layers, batch_size, self.lstm_hidden_dim)).cuda())
        else:
            h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_layers, batch_size, self.lstm_hidden_dim)))
            c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_layers, batch_size, self.lstm_hidden_dim)))

        for i in range(lstm_input.size(1)):
            ipt = lstm_input[:,i,:,:]
            ipt = torch.squeeze(ipt, 1)
            self.lstms[i].flatten_parameters()
            lstm_out, (ht, ct) = self.lstms[i](ipt, (h0, c0))
            lstm_outputs.append(ht[-1])
        lstm_outputs = torch.stack(lstm_outputs)
        lstm_outputs = lstm_outputs.transpose(1,0)
        lstm_outputs = lstm_outputs.contiguous().view(batch_size, -1)
        tag_space = self.lstm2tag(lstm_outputs)
        #print 'tag_space.size()', tag_space.size()
        tag_scores = F.log_softmax(tag_space)
        #print 'tag_scores.size()', tag_scores.size()

        return tag_scores

class LSTM_CNN(nn.Module):
    def __init__(self, lstm_input_dim, lstm_seq_len, lstm_hidden_dim, lstm2_hidden_dim, cnn_hidden_dim, tagset_size, lstm_num_layers):
        super(LSTM_CNN, self).__init__()
        self.name = 'lstm_cnn'
        self.opt = DefaultConfig()
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm2_hidden_dim = lstm2_hidden_dim
        self.lstm_seq_len = lstm_seq_len
        self.cnn_hidden_dim = cnn_hidden_dim
        self.tagset_size = tagset_size
        self.lstm_num_layers = lstm_num_layers

        self.c1_num = 16
        self.c2_num = 32
        self.c3_num = 64

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(0.5)

        if self.opt.use_gpu:
            self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, lstm_num_layers, batch_first=True, dropout=0.5).cuda()
            self.lstms2 = [nn.LSTM(6, lstm2_hidden_dim, 2, batch_first=True, dropout=0.5).cuda() for i in range(self.c3_num)]
        else:
            self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, lstm_num_layers, batch_first=True, dropout=0.5)
            self.lstms2 = [nn.LSTM(6, lstm2_hidden_dim, 2, batch_first=True, dropout=0.5) for i in range(self.c3_num)]

        #self.cnn2hidden = nn.Linear(3456, cnn_hidden_dim)
        #self.hidden2tag = nn.Linear(cnn_hidden_dim, tagset_size)
        self.hidden2tag = nn.Linear(self.lstm2_hidden_dim*self.c3_num, tagset_size)

    def forward(self, ipt):
        batch_size, window_size, feature_dim = ipt.size()
        num = window_size-self.lstm_seq_len+1
        lstm_outputs = []
        for i in range(num):
            if self.opt.use_gpu:
                h0 = Variable(torch.nn.init.orthogonal(
                    torch.Tensor(self.lstm_num_layers, batch_size, self.lstm_hidden_dim)).cuda())
                c0 = Variable(torch.nn.init.orthogonal(
                    torch.Tensor(self.lstm_num_layers, batch_size, self.lstm_hidden_dim)).cuda())
            else:
                h0 = Variable(
                    torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_layers, batch_size, self.lstm_hidden_dim)))
                c0 = Variable(
                    torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_layers, batch_size, self.lstm_hidden_dim)))
            lstm_output, (ht, ct) = self.lstm(ipt[:,i:i+self.lstm_seq_len,:], (h0, c0))
            #print 'ht[-1].size()', ht[-1].size() (batch_size, 64)
            lstm_outputs.append(ht[-1])

        lstm_outputs = torch.stack(lstm_outputs)
        cnn_inputs = lstm_outputs.unsqueeze(1)
        cnn_inputs = cnn_inputs.transpose(2, 0)
        #print 'cnn_inputs.size()', cnn_inputs.size() (6, 1, 81, 64)
        cnn_outputs =self.dropout(
            self.layer3(
                self.layer2(
                    self.layer1(
                        cnn_inputs
                    )
                )
            )
        )
        #print 'cnn_outputs.size()', cnn_outputs.size() (6, 64, 9, 6)
        lstm2_outputs = []
        for i in range(len(self.lstms2)):
            if self.opt.use_gpu:
                h0 = Variable(torch.nn.init.orthogonal(
                    torch.Tensor(self.lstm_num_layers, batch_size, self.lstm2_hidden_dim)).cuda())
                c0 = Variable(torch.nn.init.orthogonal(
                    torch.Tensor(self.lstm_num_layers, batch_size, self.lstm2_hidden_dim)).cuda())
            else:
                h0 = Variable(
                    torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_layers, batch_size, self.lstm2_hidden_dim)))
                c0 = Variable(
                    torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_layers, batch_size, self.lstm2_hidden_dim)))
            #print 'h0.size()', h0.size() (2, 6, 16)
            ipt = cnn_outputs[:,i,:,:]
            ipt = ipt.squeeze(1)
            #print 'ipt.size()', ipt.size() (6, 9, 6)
            _, (ht, ct) = self.lstms2[i](ipt, (h0, c0))
            lstm2_outputs.append(ht[-1])
        lstm2_outputs = torch.stack(lstm2_outputs)
        lstm2_outputs = lstm2_outputs.transpose(1, 0)
        #print 'lstm2_outputs.size()', lstm2_outputs.size() (6, 64, 16)
        lstm2_outputs = lstm2_outputs.contiguous().view(batch_size, -1)
        tag_scores = F.log_softmax(self.hidden2tag(lstm2_outputs))
        return tag_scores


