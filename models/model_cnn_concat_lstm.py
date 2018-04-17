import torch
from torch import nn
from ..utils import read_params
from torch.autograd import Variable
from ..config import DefaultConfig
import torch.nn.functional as F

class CNN_CONCAT_LSTM(nn.Module):
    def __init__(self, cnn_feature_dim, lstm_feature_dim, lstm_hidden_dim, tagset_size, lstm_num_hlayers):
        super(CNN_CONCAT_LSTM, self).__init__()

        self.opt = DefaultConfig()
        self.cnn_feature_dim = cnn_feature_dim
        self.lstm_feature_dim = lstm_feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.tagset_size = tagset_size
        self.lstm_num_hlayers = lstm_num_hlayers

        ipdict, portdict, = read_params()
        self.ipembed = nn.Embedding(len(ipdict)+1,30)
        self.portembed = nn.Embedding(len(portdict)+1,30)

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 10),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 10),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1344, lstm_hidden_dim)

        self.lstm = nn.LSTM(cnn_feature_dim, lstm_hidden_dim, lstm_num_hlayers, batch_first=True, dropout=0.25)

        self.lstm1 = nn.LSTMCell(lstm_feature_dim, lstm_hidden_dim)
        self.lstm2 = nn.LSTMCell(lstm_hidden_dim, lstm_hidden_dim)

        self.fc_lstm = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)

        self.hidden2tag = nn.Linear(2*lstm_hidden_dim, tagset_size)

    def forward(self, var_embed):
        """

        :param var_embed: (seq_length, window_size(100), feature_len)
        :return:
        """

        #print 'var_embed.size()', var_embed.size()
        seq_length, window_size, feature_len = var_embed.size()

        if self.opt.use_gpu:
            var_embed = var_embed.cuda()
        #print 'var_embed.size()', var_embed.size()
        out = self.mp(self.dropout(self.conv_layer3(self.conv_layer2(self.conv_layer1(
            var_embed.view(seq_length, -1, window_size, feature_len)
        )))))
        out = out.view(seq_length, -1)
        out = self.fc(out)
        out = out.view(seq_length, -1, self.lstm_hidden_dim)

        h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_hlayers, seq_length, self.lstm_hidden_dim)))
        c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_hlayers, seq_length, self.lstm_hidden_dim)))

        lstm_out, (ht, ct) = self.lstm(var_embed, (h0, c0))

        """out = out.view(-1, seq_length, self.lstm_feature_dim)
        batch_size = out.size(0)
        h0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_hlayers, batch_size, self.lstm_hidden_dim)))
        c0 = Variable(torch.nn.init.orthogonal(torch.Tensor(self.lstm_num_hlayers, batch_size, self.lstm_hidden_dim)))
        if self.opt.use_gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        lstm_out, (ht, ct) = self.lstm(out, (h0, c0))
        tag_space = self.hidden2tag(ht[-1])
        tag_scores = F.log_softmax(tag_space)
        """
        #print 'tag_space.size()', tag_space.size()
        tag_scores = F.log_softmax(tag_space)
        return tag_scores







