#!/usr/bin/ipython -i

import numpy as np
import matplotlib.pyplot as plt
import sys
caffe_root = '/u/mhauskn/projects/muupan_caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

mean_blob_path = None
image_path = 'screen/'
image_num = 1
mean_blob = None

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def PixelToRGB(pixel):
  ntsc_to_rgb = [
    0x000000, 0, 0x4a4a4a, 0, 0x6f6f6f, 0, 0x8e8e8e, 0,
    0xaaaaaa, 0, 0xc0c0c0, 0, 0xd6d6d6, 0, 0xececec, 0,
    0x484800, 0, 0x69690f, 0, 0x86861d, 0, 0xa2a22a, 0,
    0xbbbb35, 0, 0xd2d240, 0, 0xe8e84a, 0, 0xfcfc54, 0,
    0x7c2c00, 0, 0x904811, 0, 0xa26221, 0, 0xb47a30, 0,
    0xc3903d, 0, 0xd2a44a, 0, 0xdfb755, 0, 0xecc860, 0,
    0x901c00, 0, 0xa33915, 0, 0xb55328, 0, 0xc66c3a, 0,
    0xd5824a, 0, 0xe39759, 0, 0xf0aa67, 0, 0xfcbc74, 0,
    0x940000, 0, 0xa71a1a, 0, 0xb83232, 0, 0xc84848, 0,
    0xd65c5c, 0, 0xe46f6f, 0, 0xf08080, 0, 0xfc9090, 0,
    0x840064, 0, 0x97197a, 0, 0xa8308f, 0, 0xb846a2, 0,
    0xc659b3, 0, 0xd46cc3, 0, 0xe07cd2, 0, 0xec8ce0, 0,
    0x500084, 0, 0x68199a, 0, 0x7d30ad, 0, 0x9246c0, 0,
    0xa459d0, 0, 0xb56ce0, 0, 0xc57cee, 0, 0xd48cfc, 0,
    0x140090, 0, 0x331aa3, 0, 0x4e32b5, 0, 0x6848c6, 0,
    0x7f5cd5, 0, 0x956fe3, 0, 0xa980f0, 0, 0xbc90fc, 0,
    0x000094, 0, 0x181aa7, 0, 0x2d32b8, 0, 0x4248c8, 0,
    0x545cd6, 0, 0x656fe4, 0, 0x7580f0, 0, 0x8490fc, 0,
    0x001c88, 0, 0x183b9d, 0, 0x2d57b0, 0, 0x4272c2, 0,
    0x548ad2, 0, 0x65a0e1, 0, 0x75b5ef, 0, 0x84c8fc, 0,
    0x003064, 0, 0x185080, 0, 0x2d6d98, 0, 0x4288b0, 0,
    0x54a0c5, 0, 0x65b7d9, 0, 0x75cceb, 0, 0x84e0fc, 0,
    0x004030, 0, 0x18624e, 0, 0x2d8169, 0, 0x429e82, 0,
    0x54b899, 0, 0x65d1ae, 0, 0x75e7c2, 0, 0x84fcd4, 0,
    0x004400, 0, 0x1a661a, 0, 0x328432, 0, 0x48a048, 0,
    0x5cba5c, 0, 0x6fd26f, 0, 0x80e880, 0, 0x90fc90, 0,
    0x143c00, 0, 0x355f18, 0, 0x527e2d, 0, 0x6e9c42, 0,
    0x87b754, 0, 0x9ed065, 0, 0xb4e775, 0, 0xc8fc84, 0,
    0x303800, 0, 0x505916, 0, 0x6d762b, 0, 0x88923e, 0,
    0xa0ab4f, 0, 0xb7c25f, 0, 0xccd86e, 0, 0xe0ec7c, 0,
    0x482c00, 0, 0x694d14, 0, 0x866a26, 0, 0xa28638, 0,
    0xbb9f47, 0, 0xd2b656, 0, 0xe8cc63, 0, 0xfce070, 0
  ]
  rgb = ntsc_to_rgb[pixel]
  r = rgb >> 16
  g = (rgb >> 8) & 0xFF
  b = rgb & 0xFF
  return (r, g, b)

def RGBToGrayscale(rgb):
  assert(rgb[0] >= 0 and rgb[0] <= 255)
  assert(rgb[1] >= 0 and rgb[1] <= 255)
  assert(rgb[2] >= 0 and rgb[2] <= 255)
  return rgb[0] * 0.21 + rgb[1] * 0.72 + rgb[2] * 0.07

def PixelToGrayscale(pixel):
  return RGBToGrayscale(PixelToRGB(pixel))

def PreprocessScreen(raw_screen):
  kRawFrameHeight = 210;
  kRawFrameWidth = 160;
  kCroppedFrameSize = 84;
  kCroppedFrameDataSize = kCroppedFrameSize * kCroppedFrameSize;
  assert(raw_screen.shape[0] == kRawFrameHeight)
  assert(raw_screen.shape[1] == kRawFrameWidth)
  screen = np.zeros([kCroppedFrameSize, kCroppedFrameSize])
  assert(kRawFrameHeight > kRawFrameWidth)
  x_ratio = kRawFrameWidth / float(kCroppedFrameSize)
  y_ratio = kRawFrameHeight / float(kCroppedFrameSize)
  for i in xrange(kCroppedFrameSize):
    for j in xrange(kCroppedFrameSize):
      first_x = int(np.floor(j * x_ratio))
      last_x  = int(np.floor((j + 1) * x_ratio))
      first_y = int(np.floor(i * y_ratio))
      last_y  = int(np.floor((i + 1) * y_ratio))
      x_sum = 0.0
      y_sum = 0.0
      resulting_color = 0.0
      for x in xrange(first_x, last_x):
        x_ratio_in_resulting_pixel = 1.0
        if (x == first_x):
          x_ratio_in_resulting_pixel = x + 1 - j * x_ratio
        elif (x == last_x):
          x_ratio_in_resulting_pixel = x_ratio * (j + 1) - x
        assert(
            x_ratio_in_resulting_pixel >= 0.0 and
            x_ratio_in_resulting_pixel <= 1.0)
        for y in xrange(first_y, last_y):
          y_ratio_in_resulting_pixel = 1.0
          if (y == first_y):
            y_ratio_in_resulting_pixel = y + 1 - i * y_ratio
          elif (y == last_y):
            y_ratio_in_resulting_pixel = y_ratio * (i + 1) - y
          assert(
              y_ratio_in_resulting_pixel >= 0.0 and
              y_ratio_in_resulting_pixel <= 1.0)
          grayscale = RGBToGrayscale(raw_screen[int(y), int(x)])
          resulting_color += (x_ratio_in_resulting_pixel / x_ratio) * \
                             (y_ratio_in_resulting_pixel / y_ratio) * grayscale
      screen[i, j] = resulting_color
  return screen

def load_frame_data(fname):
  return np.fromfile(open(fname,'rb'), dtype=np.uint8).reshape(4,84,84).astype(np.float32)

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
  batch_size = net.blobs['frames'].data.shape[0]
  images = []
  for i in xrange(batch_size):
    fname = image_path + str(i + image_num) + '.bin'
    images.append(load_frame_data(fname))
    # images.append(PreprocessScreen(caffe.io.load_image(image_fname)))
  # data = np.asarray([preprocess('frames', im, mean=mean_blob, raw_scale=255)
  #                    for im in images])
  data = np.asarray(images)
  targets = np.zeros([batch_size,18,1,1], dtype=np.float32)
  filters = np.zeros([batch_size,18,1,1], dtype=np.float32)
  net.set_input_arrays(0, data, np.zeros([batch_size,1,1,1], dtype=np.float32))
  net.set_input_arrays(1, targets, np.zeros([batch_size,1,1,1], dtype=np.float32))
  net.set_input_arrays(2, filters, np.zeros([batch_size,1,1,1], dtype=np.float32))
  net.forward()
  image_num += batch_size

def getInputImage(num=0):
  plt.imshow(net.blobs['data'].data[num].reshape((28,28)))

def getOutputImage(num=0):
  plt.imshow(net.blobs['decodesig'].data[num].reshape((28,28)))

def showInputImage(num=0):
  getInputImage(num)
  plt.show()

def showOutputImage(num=0):
  getOutputImage(num)
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

def vis_filters(layer_num=1, num=0):
  if layer_num == 1:
    filters = net.params['conv1'][num].data
    print np.squeeze(filters).shape
    vis_square(np.squeeze(filters))
  elif layer_num == 2:
    filters = net.params['conv2'][num].data
    print filters[:20].reshape(20**2, 5, 5).shape
    vis_square(filters[:20].reshape(20**2, 5, 5))

def vis_activations(layer_name, num=0):
  feat = net.blobs[layer_name].data[0]
  print feat.shape
  vis_square(feat, padval=1)

def copy_lenet():
  lenet = caffe.Net('examples/mnist/lenet_train_test.prototxt',
                    'examples/mnist/lenet_iter_10000.caffemodel')
  np.copyto(net.params['conv1'][0].data, lenet.params['conv1'][0].data)
  np.copyto(net.params['conv1'][1].data, lenet.params['conv1'][1].data)
  np.copyto(net.params['conv2'][0].data, lenet.params['conv2'][0].data)
  np.copyto(net.params['conv2'][1].data, lenet.params['conv2'][1].data)
  np.copyto(net.params['ip1'][0].data, lenet.params['ip1'][0].data)
  np.copyto(net.params['ip1'][1].data, lenet.params['ip1'][1].data)
  np.copyto(net.params['ip2'][0].data, lenet.params['ip2'][0].data)
  np.copyto(net.params['ip2'][1].data, lenet.params['ip2'][1].data)
  net.save('examples/mnist/frankenstein.caffemodel')

# net = caffe.Net('examples/mnist/lenet_train_test.prototxt',
#                   'examples/mnist/lenet_iter_10000.caffemodel')
net = caffe.Net('examples/mnist/frankenstein.prototxt',
                'examples/mnist/deconv_iter_5000.caffemodel')
net.set_phase_test()
net.set_mode_cpu()
print 'net.blobs:'
for k, v in net.blobs.items():
  print k, v.data.shape
print 'net.params:'
for k, v in net.params.items():
  print (k, v[0].data.shape)
net.forward()
# vis_filters(2)
