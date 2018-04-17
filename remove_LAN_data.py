import json, os


if __name__ == "__main__":
    ip_dict = json.loads(open('/home/public/dsk/indexes','r').readline().strip())
    print 'len(ip_dict)', len(ip_dict)
    cand = []
    for k in ip_dict:
        if ':' in k:
            continue # to get rid of ipv6 address
        if k[:7] == '192.168' or int(k.split('.')[0])>223:
            cand.append(ip_dict[k])


    rootPath = os.path.join('/home/public/dsk/category_v4','')
    rootPath2 = os.path.join('/home/public/dsk/category_v4_noLAN','')

    for d in os.listdir(rootPath):
        dirPath = os.path.join(rootPath,d)
        dirPath2 = os.path.join(rootPath2,d)
        for f in os.listdir(dirPath):
            records = []
            with open(os.path.join(dirPath, f),'r') as fp:
                for line in fp.readlines():
                    ip = int(line.split(',')[0])
                    if not ip in cand:
                        records.append(line.strip())
            with open(os.path.join(dirPath2, f),'w') as fp:
                for record in records:
                    fp.write(record+'\n')

