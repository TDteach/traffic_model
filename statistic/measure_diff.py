from data import FlowExtractor_time
from utils import tag_to_idx
from utils import conv_to_ndarray
from config import DefaultConfig
import os

rootPaths = [os.path.join('/home/4tshare/iot/infocomm_data/category_v4', i) for i in [2,7,13,16,17,23]]
csv_paths = [os.path.join(rootPath, f) for rootPath in rootPaths for f in os.listdir(rootPath)]

opt = DefaultConfig()
data = FlowExtractor_time(csv_paths, opt.used_size, opt.capacity, opt.window_size, opt.step, filter=opt.mask)




