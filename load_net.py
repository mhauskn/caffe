import numpy as np
import matplotlib.pyplot as plt
import sys
caffe_root = './'
sys.path.insert(0, caffe_root + 'python')
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

image_num = 0
mean_blob = None

def preprocess(input_name, input_, mean=None, input_scale=None,
               raw_scale=None, channel_order=None):
  caffe_in = input_.astype(np.float32, copy=False)
  in_size = net.blobs[input_name].data.shape[2:]
  if caffe_in.shape[:2] != in_size:
      caffe_in = caffe.io.resize_image(caffe_in, in_size)
  if channel_order is not None:
      caffe_in = caffe_in[:, :, channel_order]
  caffe_in = caffe_in.transpose((2, 0, 1))
  if raw_scale is not None:
      caffe_in *= raw_scale
  if mean is not None:
      caffe_in -= mean
  if input_scale is not None:
      caffe_in *= input_scale
  return caffe_in

def deprocess(input_, mean=None, input_scale=None,
               raw_scale=None, channel_order=None):
  decaf_in = input_.copy().squeeze()
  if input_scale is not None:
      decaf_in /= input_scale
  if mean is not None:
      decaf_in += mean
  if raw_scale is not None:
      decaf_in /= raw_scale
  decaf_in = decaf_in.transpose((1,2,0))
  if channel_order is not None:
      channel_order_inverse = [channel_order.index(i)
                               for i in range(decaf_in.shape[2])]
      decaf_in = decaf_in[:, :, channel_order_inverse]
  return decaf_in

def load_mean_blob():
  global mean_blob
  blob = caffe.proto.caffe_pb2.BlobProto()
  blob.ParseFromString(open('examples/atari/mean.binaryproto').read())
  mean_blob = caffe.io.blobproto_to_array(blob)[0]
  # plt.imshow(np.rollaxis(mean_blob,0,3))
  # plt.show()

def run_forward():
  global image_num
  batch_size = net.blobs['data'].data.shape[0]
  images = []
  for i in xrange(batch_size):
    images.append(caffe.io.load_image('examples/images/atari'+str(i+image_num)+'.jpg'))
  data = np.asarray([preprocess('data', im, mean=mean_blob, raw_scale=255) for im in images])
  labels = np.zeros([batch_size,1,1,1], dtype=np.float32)
  net.set_input_arrays(data, labels)
  net.forward()
  image_num += batch_size

def getInputImage(num=0):
  plt.imshow(deprocess(net.blobs['data'].data[num], mean=mean_blob, raw_scale=255))

def showInputImage(num=0):
  getInputImage(num)
  plt.show()

def showInputImageActivation(num=0):
  getInputImage(num)
  plt.annotate(str(net.blobs['ip2'].data[num]),
               xy=(0,0), xytext=(.8,0), fontsize=20)
  plt.show()

def SaveImagesAndActivations():
  for i in range(net.blobs['data'].data.shape[0]):
    getInputImage(i)
    text = 'Min: ' + str(np.min(net.blobs['ip2'].data[i])) + \
           ' Avg: ' + str(np.mean(net.blobs['ip2'].data[i])) + \
           ' Max: ' + str(np.max(net.blobs['ip2'].data[i]))
    plt.annotate(text, xy=(0,0.01), xycoords='axes fraction',
                 fontsize=12, color='w')
    plt.savefig('figs/%d.jpg'%i)
    plt.clf()
    plt.close()
    print i

def findNonzeroLabel():
  while not np.nonzero(net.blobs['label'].data)[0]:
    print 'all zero'
    net.forward()

# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
  data -= data.min()
  data /= data.max()
  # force the number of filters to be square
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = ((0, n ** 2 - data.shape[0]),
             (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
  data = np.pad(data, padding, mode='constant',
                constant_values=(padval, padval))
  # tile the filters into an image
  data = data.reshape(
    (n, n) + data.shape[1:]).transpose(
      (0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
  plt.imshow(data)
  plt.show()

def vis_filters(layer_num=1):
  if layer_num == 1:
    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1))
  elif layer_num == 2:
    filters = net.params['conv2'][0].data
    vis_square(filters[:32].reshape(32**2, 5, 5))

def vis_feats(layer_num=1):
  feat = net.blobs['conv'+str(layer_num)].data[0]
  vis_square(feat, padval=1)


if len(sys.argv) < 3:
  raise Exception('usage: ipython -i load_net.py '\
                  'examples/atari_train_test.proto '\
                  'examples/atari/iter_n.caffemodel')
else:
  net = caffe.Net(sys.argv[1], sys.argv[2])
  net.set_phase_test()
  net.set_mode_cpu()
  print 'net.blobs:'
  for k, v in net.blobs.items():
    print k, v.data.shape
  print 'net.params:'
  for k, v in net.params.items():
    print (k, v[0].data.shape)
  load_mean_blob()
  run_forward()
