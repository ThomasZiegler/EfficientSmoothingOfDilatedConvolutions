import tensorflow as tf
import numpy as np
import six
import sys



"""
This script provides different 2d dilated convolutions.

I appreciate ideas for a more efficient implementation of the proposed two smoothed dilated convolutions.
"""



def _dilated_conv2d(dilated_type, x, kernel_size, num_o, dilation_factor, name,
                    filter_size=1, biased=False):
    if dilated_type == 'regular':
        return _regular_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, filter_size, biased)
    elif dilated_type == 'decompose':
        return _decomposed_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, filter_size, biased)
    elif dilated_type == 'smooth_GI':
        return _smoothed_dilated_conv2d_GI(x, kernel_size, num_o, dilation_factor, name, filter_size, biased)
    elif dilated_type == 'smooth_SSC':
        return _smoothed_dilated_conv2d_SSC(x, kernel_size, num_o, dilation_factor, name, filter_size, biased)
    elif dilated_type == 'average_filter':
        return _averaged_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, filter_size, biased)
    elif dilated_type == 'gaussian_filter':
        return _gaussian_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, filter_size, biased)


    else:
        print('dilated_type ERROR!')
        print("Please input: regular, decompose, smooth_GI or smooth_SSC")
        sys.exit(-1)

def _regular_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name,
                            filter_size=1, biased=False):
    """
    Dilated conv2d without BN or relu.
    """
    num_x = x.shape[3].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o

def _averaged_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, filter_size=1, biased=False):
    """
    Dilated conv2d with antecedent average filter and without BN or relu.
    """
    num_x = x.shape[3].value

    # perform averaging (as seprable convolution)
    w_avg_value = 1.0/(filter_size*filter_size)
    w_avg = tf.Variable(tf.constant(w_avg_value,
                                    shape=[filter_size,filter_size,num_x,1]), name='w_avg',trainable=False)
    o = tf.nn.depthwise_conv2d_native(x, w_avg, [1,1,1,1], padding='SAME')
#    o = tf.expand_dims(x, -1)
#    o = tf.nn.conv3d(o, w_avg, strides=[1,1,1,1,1], padding='SAME')
#    o = tf.squeeze(o, -1)

    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o



def _combinational_layer(x, kernel_size, num_o, dilation_factor, name, filter_size=1, biased=False):
    """
    Dilated conv2d with antecedent average filter and without BN or relu.
    """
    num_x = x.shape[3].value
    
    
    #Vector with 4 elements
    #1st corresponding to identity
    #2nd to regular convolution
    #3th to averaging filter
    #4th to gaussian filter
    
    c_ = tf.Variable([0,0,0,0],name="c_vector")
    #this needs to be trainable...
    o1 = x
    o2 = tf.nn.conv2d(x,[filter_size,filter_size,num_x,num_x],[1,1,1,1],padding="SAME",name="w_conv")

    # perform averaging (as seprable convolution)
    w_avg_value = 1.0/(filter_size*filter_size)
    w_avg = tf.Variable(tf.constant(w_avg_value,
                                    shape=[filter_size,filter_size,num_x,1]), name='w_avg',trainable=False)
    o3 = tf.nn.depthwise_conv2d_native(x, w_avg, [1,1,1,1], padding='SAME')
#    o = tf.expand_dims(x, -1)
#    o = tf.nn.conv3d(o, w_avg, strides=[1,1,1,1,1], padding='SAME')
#    o = tf.squeeze(o, -1)



    # perform gaussian filtering (as seprable convolution)
    sigma = tf.constant(1.0, shape=[1])
    # create kernel grid
    ax = tf.range(-filter_size//2+1, filter_size//2+1, dtype=tf.float32)
    xx, yy = tf.meshgrid(ax, ax)
    # calculate weight and reshape to correct shape
    w_gauss_value = tf.exp(-(xx**2 + yy**2) / (2.*sigma**2))
    w_gauss_value = w_gauss_value / tf.reduce_sum(w_gauss_value)

    # dublicate kernel num_x times
    w_gauss_value = tf.tile(tf.expand_dims(w_gauss_value,-1), [1,1,num_x])
    # add expand one dimension to match depthwise_conv2d_native filter requirement
    w_gauss_value = tf.expand_dims(w_gauss_value,-1)
    w_gauss = tf.Variable(w_gauss_value, name='w_gauss',trainable=False)

    
    o4 = tf.nn.depthwise_conv2d_native(x, w_gauss, [1,1,1,1], padding='SAME')
    
    c_ = tf.nn.softmax(c_)
    o = c[0]*o1+c[1]*o2+c[2]*o3+c[3]*o4
       
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o


def _gaussian_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, filter_size=1, biased=False):
    """
    Dilated conv2d with antecedent gaussian filter and without BN or relu.
    """
    num_x = x.shape[3].value

    # perform gaussian filtering (as seprable convolution)
    sigma = tf.constant(1.0, shape=[1])
    # create kernel grid
    ax = tf.range(-filter_size//2+1, filter_size//2+1, dtype=tf.float32)
    xx, yy = tf.meshgrid(ax, ax)
    # calculate weight and reshape to correct shape
    w_gauss_value = tf.exp(-(xx**2 + yy**2) / (2.*sigma**2))
    w_gauss_value = w_gauss_value / tf.reduce_sum(w_gauss_value)

    # dublicate kernel num_x times
    w_gauss_value = tf.tile(tf.expand_dims(w_gauss_value,-1), [1,1,num_x])
    # add expand one dimension to match depthwise_conv2d_native filter requirement
    w_gauss_value = tf.expand_dims(w_gauss_value,-1)
    w_gauss = tf.Variable(w_gauss_value, name='w_gauss',trainable=False)

    
    o = tf.nn.depthwise_conv2d_native(x, w_gauss, [1,1,1,1], padding='SAME')

    #o = tf.expand_dims(x, -1)
    #o = tf.nn.conv3d(o, w_gauss, strides=[1,1,1,1,1], padding='SAME')
    #o = tf.squeeze(o, -1)

    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o

def _decomposed_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, filter_size=1, biased=False):
    """
    Decomposed dilated conv2d without BN or relu.
    """
    # padding so that the input dims are multiples of dilation_factor
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    pad_bottom = (dilation_factor - H % dilation_factor) if H % dilation_factor != 0 else 0
    pad_right = (dilation_factor - W % dilation_factor) if W % dilation_factor != 0 else 0
    pad = [[0, pad_bottom], [0, pad_right]]
    # decomposition to smaller-sized feature maps
    # [N,H,W,C] -> [N*d*d, H/d, W/d, C]
    o = tf.space_to_batch(x, paddings=pad, block_size=dilation_factor)
    # perform regular conv2d
    num_x = x.shape[3].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        s = [1, 1, 1, 1]
        o = tf.nn.conv2d(o, w, s, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
    o = tf.batch_to_space(o, crops=pad, block_size=dilation_factor)
    return o

def _smoothed_dilated_conv2d_GI(x, kernel_size, num_o, dilation_factor, name, filter_size=1, biased=False):
    """
    Smoothed dilated conv2d via the Group Interaction (GI) layer without BN or relu.
    """
    # padding so that the input dims are multiples of dilation_factor
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    pad_bottom = (dilation_factor - H % dilation_factor) if H % dilation_factor != 0 else 0
    pad_right = (dilation_factor - W % dilation_factor) if W % dilation_factor != 0 else 0
    pad = [[0, pad_bottom], [0, pad_right]]
    # decomposition to smaller-sized feature maps
    # [N,H,W,C] -> [N*d*d, H/d, W/d, C]
    o = tf.space_to_batch(x, paddings=pad, block_size=dilation_factor)
    # perform regular conv2d
    num_x = x.shape[3].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        s = [1, 1, 1, 1]
        o = tf.nn.conv2d(o, w, s, padding='SAME')
        fix_w = tf.Variable(tf.eye(dilation_factor*dilation_factor), name='fix_w')
        l = tf.split(o, dilation_factor*dilation_factor, axis=0)
        os = []
        for i in six.moves.range(0, dilation_factor*dilation_factor):
            os.append(fix_w[0, i] * l[i])
            for j in six.moves.range(1, dilation_factor*dilation_factor):
                os[i] += fix_w[j, i] * l[j]
        o = tf.concat(os, axis=0)
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
    o = tf.batch_to_space(o, crops=pad, block_size=dilation_factor)
    return o

def _smoothed_dilated_conv2d_SSC(x, kernel_size, num_o, dilation_factor, name, filter_size=1, biased=False):
    """
    Smoothed dilated conv2d via the Separable and Shared Convolution (SSC) without BN or relu.
    """
    num_x = x.shape[3].value
    fix_w_size = dilation_factor * 2 - 1
    with tf.variable_scope(name) as scope:
        fix_w = tf.get_variable('fix_w', shape=[fix_w_size, fix_w_size, 1, 1, 1], initializer=tf.zeros_initializer)
        mask = np.zeros([fix_w_size, fix_w_size, 1, 1, 1], dtype=np.float32)
        mask[dilation_factor - 1, dilation_factor - 1, 0, 0, 0] = 1
        fix_w = tf.add(fix_w, tf.constant(mask, dtype=tf.float32))
        o = tf.expand_dims(x, -1)
        o = tf.nn.conv3d(o, fix_w, strides=[1,1,1,1,1], padding='SAME')
        o = tf.squeeze(o, -1)
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o
