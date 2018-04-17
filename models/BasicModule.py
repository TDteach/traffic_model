import torch
from torch import nn
import time

class BasicModule(nn.Module):
    """
    A basic wrapping of nn.Module, mainly provides function 'save' and 'load'
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        save model parameters, name-->"model+time.pth"
        :param name:
        :return:
        """
        if name is None:
            prefix = '/home/4tshare/iot/dev/traffic_model/checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix+'%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name
