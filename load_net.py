import numpy as np
import matplotlib.pyplot as plt
import sys
caffe_root = './'
sys.path.insert(0, caffe_root + 'python')
import caffe

if len(sys.argv) < 3:
    print 'usage:', sys.argv[0], 'net.proto net.caffemodel'
    exit()

net = caffe.Net(sys.argv[1], sys.argv[2])
net.set_phase_train()
net.set_mode_cpu()

def showInputImage(num=0, chan=0):
    plt.imshow(net.blobs['data'].data[num][chan])
    plt.show()

def showInputImageActivation(num=0):
    plt.imshow(net.blobs['data'].data[num][0])
    plt.annotate(str(net.blobs['ip2'].data[num]),
                 xy=(0,0), xytext=(.8,0), fontsize=20)
    plt.show()

def SaveImagesAndActivations():
    net.forward()
    for i in range(net.blobs['data'].data.shape[0]):
        plt.imshow(net.blobs['data'].data[i][0])
        text = 'Min: ' + str(np.min(net.blobs['ip2'].data[i])) + \
               ' Avg: ' + str(np.mean(net.blobs['ip2'].data[i])) + \
               ' Max: ' + str(np.max(net.blobs['ip2'].data[i]))
        plt.annotate(text, xy=(0,0.01), xycoords='axes fraction', fontsize=12)
        plt.savefig('figs/%d.pdf'%i)
        plt.clf()
        plt.close()

def findNonzeroLabel():
    while not np.nonzero(net.blobs['label'].data)[0]:
        print 'all zero'
        net.forward()
