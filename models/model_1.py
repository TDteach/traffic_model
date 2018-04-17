import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from ..utils import read_params
from BasicModule import BasicModule

class model_1(BasicModule):

    def __init__(self, feature_dim, hidden_dim, tagset_size, num_layers, cuda_support=True):
        super(model_1, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cuda_support = cuda_support

        ipdict, portdict, protdict = read_params()
        self.ipembed = nn.Embedding(len(ipdict)+1,10)
        self.portembed = nn.Embedding(len(portdict)+1,10)
        self.protembed = nn.Embedding(len(protdict)+1,10)
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
    def init_hidden(self, x):
        """
        Note that if you use bidirectional=1 in lstm, each hidden layer should be (2,1,hidden_dim)
        :return:
        """
        return (autograd.Variable(nn.init.orthogonal(torch.Tensor(self.num_layers*2,x,self.hidden_dim)).cuda()),
                autograd.Variable(nn.init.orthogonal(torch.Tensor(self.num_layers*2,x,self.hidden_dim)).cuda()))

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
            lstm_out, _ = self.lstm(var_embed, (h0, c0))
            #print 'lstm_out.size() = ', lstm_out.size() # (batch_size,window_size,hidden_dim*2)
            #print 'lstm_out', lstm_out
            tag_space = self.hidden2tag(lstm_out)
            #print 'tag_space', tag_space

            #Decode hidden state of last time step
            tag_scores = F.log_softmax(tag_space[:,-1,:])
            return tag_scores


