from dataset import Traffic
from torch.utils.data import DataLoader

traffic = Traffic('/home/4tshare/iot/infocomm_data/category/16-09-23/1.csv')
dataloader = DataLoader(traffic, batch_size=4, shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print i_batch, sample_batched['window'].size()
    print sample_batched
print traffic[0]

