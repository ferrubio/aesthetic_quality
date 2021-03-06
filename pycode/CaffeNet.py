import numpy as np
import caffe
import tempfile
import os
from caffe.proto import caffe_pb2
from caffe import layers as L
from caffe import params as P

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write((str(n.to_proto())).encode())
        return f.name
    
def VGG16(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying VGG-16, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 3, 64, pad=1, param=param)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 3, 64, pad=1, param=param)
    
    n.pool1 = max_pool(n.relu1_2, 2, stride=2)
    
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 3, 128, pad=1, param=param)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 3, 128, pad=1, param=param)
    
    n.pool2 = max_pool(n.relu2_2, 2, stride=2)
    
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 3, 256, pad=1, param=param)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 3, 256, pad=1, param=param)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 3, 256, pad=1, param=param)
    
    n.pool3 = max_pool(n.relu3_3, 2, stride=2)
    
    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 3, 512, pad=1, param=param)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 3, 512, pad=1, param=param)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 3, 512, pad=1, param=param)
    
    n.pool4 = max_pool(n.relu4_3, 2, stride=2)
    
    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 3, 512, pad=1, param=param)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 3, 512, pad=1, param=param)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 3, 512, pad=1, param=param)
    
    n.pool5 = max_pool(n.relu5_3, 2, stride=2)
    
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write((str(n.to_proto())).encode())
        return f.name
    
    
def caffenet_net(train=True, learn_all=False, subset=None, source_path=''):
    if subset is None:
        subset = 'train' if train else 'test'
        
    source = source_path % subset
    caffe_root = '/opt/caffe/'
    
    transform_param = dict(mirror=train, crop_size=227,
        mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=128, new_height=256, new_width=256, ntop=2)
    return caffenet(data=style_data, label=style_label, train=train,num_classes=2,classifier_name='fc8_aesthetic', learn_all=learn_all)


def VGG16_net(train=True, learn_all=False, subset=None, source_path=''):
    if subset is None:
        subset = 'train' if train else 'test'
    
    source = source_path % subset
    caffe_root = '/opt/caffe/'
    
    transform_param = dict(mirror=train, crop_size=224,
        mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=256, new_height=256, new_width=256, ntop=2)
    return VGG16(data=style_data, 
                     label=style_label, 
                     train=train,
                     num_classes=2,
                     classifier_name='fc8_aesthetic', 
                     learn_all=learn_all)
    

def solver(train_net_path, test_net_path=None, base_lr=0.001, snapshot_pref=''):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 30000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 100

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    s.snapshot_prefix = snapshot_pref
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write((str(s)).encode())
        return f.name
    
def run_solvers(niter, solvers, disp_interval=1000):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print ('%3d) %s' % (it, loss_disp))     
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights

'''
def eval_aest_net(weights, test_iters=10):
    test_net = caffe.Net(aest_net(train=False), weights, caffe.TEST)
    accuracy = 0
    for it in range(0,test_iters):
        accuracy += test_net.forward()['acc']
    accuracy /= test_iters
    return test_net, accuracy
'''
def get_conf_matrix(train_net,list_images,layer,net,transformer):
    # set the size of the input (we can skip this if we're happy
    # with the default; we can also change it later, e.g., for different batch sizes)
    batch, C, H, W = net.blobs['data'].shape
    
    num_images = len(list_images)
    
    output_shape = net.blobs[layer].shape
    output_shape[0] = num_images
    output = np.zeros(output_shape)
    
    count = 0
    while count < num_images:
        pre_count = count    
        for i in range(0,batch):
            if count >= num_images:
                i -= 1
                break
            try:
                image = caffe.io.load_image(images_root + list_images[count])
                if len(image.shape)>3:
                    image = image[0]
                transformed_image = transformer.preprocess('data', image)
                net.blobs['data'].data[i] = transformed_image
            except:
                print (count)
                raise
            count += 1
        
        net.forward()
    
        output[pre_count:count]=net.blobs[layer].data[0:i+1]
    
    return output