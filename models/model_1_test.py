from model_1 import model_1
from torch.utils.data import DataLoader
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..data import Traffic
from ..data import Dispatcher
from ..utils import tag_to_idx

model = model_1(33, 16, 16, cuda_support=True).cuda()
loss_function = nn.NLLLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1)


#traffic = Traffic('/home/4tshare/iot/infocomm_data/category/16-09-23/1.csv')
traffic = Dispatcher(['/home/4tshare/iot/infocomm_data/category/16-09-23/1.csv',
                      '/home/4tshare/iot/infocomm_data/category/16-09-23/2.csv',])
dataloader = DataLoader(traffic, batch_size=4, shuffle=True, num_workers=4)

print traffic.stat()
for i_batch, sample_batched in enumerate(dataloader):
    model.zero_grad()

    sample_batched_tensor = sample_batched['window'].type(torch.cuda.FloatTensor)
    #print sample_batched['window']
    #print sample_batched['label']

    model.hidden = model.init_hidden(sample_batched_tensor.size()[0])
    tag_scores = model(sample_batched_tensor)
    print 'tag_scores.size() = ', tag_scores.size()
    last_output = tag_scores[:,-1,:]
    print 'last_output.size() = ', last_output.size()
    #loss = loss_function(tag_scores, autograd.Variable(sample_batched_label_tensor))
    loss = loss_function(last_output, autograd.Variable(torch.cuda.LongTensor([tag_to_idx(e) for e in sample_batched['label']])))
    loss.backward()
    optimizer.step()
