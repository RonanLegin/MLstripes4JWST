import numpy as np 
import tensorflow as tf


### Unet parameters ###

stacks = 5 # number of downsampling blocks
stackwidth = 1 # number of convolution layers per block
filters_base = 24 # number of kernels convolved per convolution layer
kernel_size = 7 # size of kernels

kw = {}
kw['activation'] = tf.nn.relu # non-linear activation function 
kw['padding'] = 'SAME' # type of padding 


### Training parameters ###

# size of gradient step to update network weights
learning_rate = 5e-5
# number of training examples fed for every weight optimization
batch_size = 8
# Total number of optimization steps to train the network
num_steps = 100000


# Number of pixels on height and width for COSMOS training data used.
cosmos_npix = 192
