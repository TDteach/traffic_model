import os
import pandas as pd

rootPath = os.path.join('','/home/4tshare/iot/infocomm_data')
catPath = os.path.join(rootPath,'category_v3')

names = ['ip','port','ipv4','ipv6','tcp','udp','http','ssl','dns','direction','datalen','timeval','tag']





if __name__ == "__main__":
    for f in os.listdir(rootPath):
        if f[-11:] == '_v3_new.csv':
            time = f[:-11]
            csv = pd.read_csv(os.path.join(rootPath,f), header=None, names=names)
            if not os.path.exists(os.path.join(catPath,time)):
                os.mkdir(os.path.join(catPath,time))
            timePath = os.path.join(catPath, time)

            devices = set(csv['tag'])
            for dev in devices:
                df = csv[csv['tag'] == dev]
                df.to_csv(os.path.join(timePath,str(dev)+'.csv'),header=False,index=False)
