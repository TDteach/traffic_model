import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

from ..utils import read_params
from BasicModule import BasicModule

class model_4(BasicModule):
    """
    In this model, I used newly-generated data in /home/4tshare/iot/infocomm_data/category_v3
    It has ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','datalen','timeval'] features.
    Due to tcp has a hand-shaking procedure, I checked the PSH flag in tcp header to get the tcp packets really transferring data instead of those establishing connections.

    Also some other data cleaning work should be done to remove those packets only existing in LAN environments. like ARP packets.
    """
    def __init__(self, feature_dim, hidden_dim, tagset_size, num_layers, cuda_support=True):
        super(model_4, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cuda_support = cuda_support

        ipdict, portdict, = read_params()
        self.ipembed = nn.Embedding(len(ipdict)+1,30)
        self.portembed = nn.Embedding(len(portdict)+1,30)
        #self.protembed = nn.Embedding(len(protdict)+1,10)
        """
        nn.Embedding can only handle autograd.Variable
        """
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
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
        h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers*2, var_embed.size()[0], self.hidden_dim)).cuda())
        c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.num_layers*2, var_embed.size()[0], self.hidden_dim)).cuda())

        #print 'vec_seq', vec_seq (batch_size, window_size, feature_len)
        if self.cuda_support:
            #print 'vec_seq[:,:,0].type(torch.cuda.LongTensor)', vec_seq[:,:,0].type(torch.cuda.LongTensor)
            #print 'ipe', ipe#, ipe (batch_size, window_size, 10)
            var_embed.cuda()
            #print 'var_embed', var_embed
            #print 'var_embed', var_embed (batch_size, window_size, 33)
            #lstm_out, self.hidden = self.lstm(var_embed, self.hidden)
            packed_out, (ht, ct) = self.lstm(var_embed, (h0, c0))
            #print 'lstm_out.size() = ', lstm_out.size() # (batch_size,window_size,hidden_dim*2)
            #print 'lstm_out', lstm_out
            print 'ht.size()', ht.size()
            print 'ct.size()', ct.size()
            print 'packed_out.size()', packed_out.size()
            lstm_output, _ = pad_packed_sequence(packed_out)

            tag_space = self.hidden2tag(ht[-1])
            #print 'tag_space', tag_space

            #Decode hidden state of last time step
            tag_scores = F.log_softmax(tag_space[:,-1,:])
            return tag_scores


