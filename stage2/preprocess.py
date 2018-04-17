import os
import labels
from contextlib import nested
from optparse import OptionParser

def parseCSV(csvin, csvout, devmac, label):
    sigs = []
    with nested(open(csvin,'r'), open(csvout,'w')) as (f1,f2):
        for line in f1.readlines():
            try:
                elems = line.strip().split(',')
                srcIP = elems[0]
                dstIP = elems[1]
                srcPort = elems[2]
                dstPort = elems[3]
                ipv4, ipv6, tcp, udp, http, ssl, dns = elems[4], elems[5], elems[6], elems[7], elems[8], elems[9], elems[10]
                srcMac = elems[11]
                dstMac = elems[12]
                datalen = elems[13]
                timeval = elems[-1]
            except Exception:
                continue
# ip, port, ipv4, ipv6, tcp, udp, http, ssl, dns, direction, dataLen, timeval, tag = line.strip().split(',')
            if srcIP != 'Unknown' and dstIP != 'Unknown':
                if srcMac == devmac:
                    if dstIP[:7] != '192.168' and int(dstIP.split('.')[0])<=223:
                    #if not (int(dstIP.split('.')[0])>=192 and int(dstIP.split('.')[0])<=223):
                        sig = ','.join([dstIP, dstPort, ipv4, ipv6, tcp, udp, http, ssl, dns, '0', datalen, timeval])
                        sig += ','+str(label)
                        sigs.append(sig)
                elif dstMac == devmac:
                    if srcIP[:7] != '192.168' and int(srcIP.split('.')[0])<=223:
                    #if not (int(srcIP.split('.')[0])>=192 and int(srcIP.split('.')[0])<=223):
                        sig = ','.join([dstIP, dstPort, ipv4, ipv6, tcp, udp, http, ssl, dns, '1', datalen, timeval])
                        sig += ','+str(label)
                        sigs.append(sig)

        for sig in sigs:
            f2.write(sig+'\n')

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input", action="store", dest="csvin")
    parser.add_option("-o", "--output", action="store", dest="csvout")
    parser.add_option("-d", "--device", action="store", dest="dev")
    parser.add_option("-l", "--label", action="store", dest="label")

    (options, args) = parser.parse_args()
    devices = labels.devices
    devmac = devices[options.dev]
    parseCSV(options.csvin, options.csvout, devmac, options.label)








