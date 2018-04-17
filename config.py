import warnings
import os
import random
class DefaultConfig(object):
    env = 'default'
    model = 'model_4'

    train_data_roots = ['/home/public/dsk/category_v3/16-09-23',
                        '/home/public/dsk/category_v3/16-09-24',
                        '/home/public/dsk/category_v3/16-09-25',
                        '/home/public/dsk/category_v3/16-09-26',
                        '/home/public/dsk/category_v3/16-09-27',
                        '/home/public/dsk/category_v3/16-09-28',
                        '/home/public/dsk/category_v3/16-09-29',
                        '/home/public/dsk/category_v3/16-09-30',
                        '/home/public/dsk/category_v3/16-10-01',
                        '/home/public/dsk/category_v3/16-10-02',
                        '/home/public/dsk/category_v3/16-10-03',
                        '/home/public/dsk/category_v3/16-10-04',
                        '/home/public/dsk/category_v3/16-10-05',
                        '/home/public/dsk/category_v3/16-10-06',
                        '/home/public/dsk/category_v3/16-10-07',
                        '/home/public/dsk/category_v3/16-10-08',
                        '/home/public/dsk/category_v3/16-10-09',
                        '/home/public/dsk/category_v3/16-10-10',
                        '/home/public/dsk/category_v3/16-10-12',
                        '/home/public/dsk/category_v3/16-10-11']

    csv_paths = [os.path.join(r,f) for r in train_data_roots for f in os.listdir(r)]

    load_model_path = 'checkpoints/model.pth'

    tag_set_size = 30
    batch_size = 6
    cnn_lstm_seq_len = 10
    cnn_lstm_win_size = 100
    use_gpu = True
    num_workers = 4
    print_freq = 20
    window_size = 100#10
    step = random.randint(10, 60)
    capacity = 20000

    used_size = 40000
    mask = ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction1','direction2','datalen','timeinterval','tag']
    #mask1 = ['direction1', 'direction2', 'datalen', 'timeinterval', 'tag']

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 4
    lr = 0.01
    lr_decay = 0.95
    weight_decay = 1e-4

    def parse(self, kwargs):
        for k,v in kwargs.iteritems():
            if not hasattr(self, k):
                warnings.warn("Warning: config has not attribute %s"%k)
            setattr(self,k,v)
        print 'user config:'
        for k,v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print k, getattr(self,k)
