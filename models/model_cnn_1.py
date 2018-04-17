import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from ..utils import read_params
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, feature_dim, tagset_size, embed_dim):
        super(CNN, self).__init__()
        ipdict, portdict = read_params()
        self.name = 'cnn_'+str(feature_dim)
        self.feature_dim = feature_dim
        self.ipembed = nn.Embedding(len(ipdict)+1,embed_dim)
        self.portembed = nn.Embedding(len(portdict)+1,embed_dim)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 10),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, stride=2, padding=1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 10),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1)
        )
        #self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, tagset_size)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.mp(out)
        out = self.dropout(out)
        #print 'out.size()', out.size()
        out = out.view(in_size, -1)
        out = self.fc(out)
        tag_scores = F.log_softmax(out)
        #print 'tag_scores.size()', tag_scores.size()
        return tag_scores

class CNN2(nn.Module):

    def __init__(self, window_size, feature_dim, hidden_dim, tagset_size, embed_dim):
        super(CNN2, self).__init__()
        ipdict, portdict = read_params()
        self.name = 'cnn_'+str(feature_dim)+'_'+str(hidden_dim)
        self.window_size = window_size
        self.feature_dim = feature_dim
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
        self.fc2hidden = nn.Linear(448, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer3(self.layer2(self.layer1(x)))
        out = self.dropout(out)
        out = out.view(in_size, -1)
        out = self.fc2hidden(out)
        out = self.hidden2tag(out)
        tag_scores = F.log_softmax(out)
        return tag_scores

class CNN_noip_noport_noprot(nn.Module):
    def __init__(self, feature_dim, embed_dim):
        super(CNN_noip_noport_noprot, self).__init__()
        ipdict, portdict = read_params()
        self.feature_dim = feature_dim
        self.ipembed = nn.Embedding(len(ipdict)+1,embed_dim)
        self.portembed = nn.Embedding(len(portdict)+1,embed_dim)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(768, 30)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer1(x)
        out = self.layer2(out)
        #out = self.layer3(out)
        #out = self.mp(out)
        out = self.dropout(out)
        #print 'out.size()', out.size()
        out = out.view(in_size, -1)
        out = self.fc(out)
        tag_scores = F.log_softmax(out)
        #print 'tag_scores.size()', tag_scores.size()
        return tag_scores

class CNN_onlymac(nn.Module):
    # for small stacked
    def __init__(self, tagset_size):
        super(CNN_onlymac, self).__init__()
        self.name = 'cnn_onlymac'
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        #self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(192, tagset_size)

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        in_size = x.size(0)
        out = self.layer1(x)
        #print 'layer1out.size()', out.size()
        out = self.layer2(out)
        #print 'layer2out.size()', out.size()
        #out = self.layer3(out)
        #out = self.mp(out)
        out = self.dropout(out)
        #print 'out.size()', out.size()
        out = out.view(in_size, -1)
        out = self.fc(out)
        tag_scores = F.log_softmax(out)
        #print 'tag_scores.size()', tag_scores.size()
        return tag_scores