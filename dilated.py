import tensorflow as tf
import numpy as np
import six
import sys



"""
This script provides different 2d dilated convolutions.

I appreciate ideas for a more efficient implementation of the proposed two smoothed dilated convolutions.
"""



def _dilated_conv2d(dilated_type, x, kernel_size, num_o, dilation_factor, name,
                    top_scope, biased=False):
    if dilated_type == 'regular':
        return _regular_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)
    elif dilated_type == 'decompose':
        return _decomposed_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)
    elif dilated_type == 'smooth_GI':
        return _smoothed_dilated_conv2d_GI(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)
    elif dilated_type == 'smooth_SSC':
        return _smoothed_dilated_conv2d_SSC(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)
    elif dilated_type == 'average_filter':
        return _averaged_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)
    elif dilated_type == 'gaussian_filter':
        return _gaussian_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)
    elif dilated_type == 'combinational_layer':
        return _combinational_layer(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)

    else:
        print('dilated_type ERROR!')
        print("Please input: regular, decompose, smooth_GI or smooth_SSC")
        sys.exit(-1)

def _regular_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name,
                            top_scope=1, biased=False):
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

def _averaged_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope=1, biased=False):
    """
    Dilated conv2d with antecedent average filter and without BN or relu.
    """
    num_x = x.shape[3].value

    filter_size = dilation_factor - 1

    # perform averaging (as seprable convolution)
    w_avg_value = 1.0/(filter_size*filter_size)
    w_avg = tf.Variable(tf.constant(w_avg_value,
                                    shape=[filter_size,filter_size,num_x,1]), name='w_avg',trainable=False)
    o = tf.nn.depthwise_conv2d_native(x, w_avg, [1,1,1,1], padding='SAME')

    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o


def _c_vector(o_avg, o_gauss, o_ssc):
    c_values = [0.33, 0.33, 0.33]
    c_init = tf.constant_initializer(c_values)
    c_ = tf.get_variable('c_vector', shape=[3], initializer=c_init)
    c_sum = tf.reduce_sum(c_)
    c_.assign( c_ / c_sum)
    o = c_[0]*o_avg + c_[1]*o_gauss + c_[2]*o_ssc

    return o 


def _gauss_sigma():
    sigma_init = 1.0
    init = tf.constant_initializer(sigma_init)
    sigma = tf.get_variable('gauss_sigma', shape=[1], initializer=init)

    return sigma


# use only one sigma for all dilated convolutions
#def _combinational_layer(x, kernel_size, num_o, dilation_factor, name, top_scope=1, biased=False):
#    """
#    Dilated conv2d with antecedent average filter and without BN or relu.
#    """
#    num_x = x.shape[3].value
#    filter_size = dilation_factor - 1
#    fix_w_size = dilation_factor * 2 - 1
#
#    c_vector = tf.make_template('const_vector', _c_vector)
#    gauss_sigma = tf.make_template('sigma', _gauss_sigma)
#
#    # perform averaging (as seprable convolution)
#    w_avg_value = 1.0/(filter_size*filter_size)
#    w_avg = tf.Variable(tf.constant(w_avg_value,
#                                    shape=[filter_size,filter_size,num_x,1]), name='w_avg',trainable=False)
#    o_avg = tf.nn.depthwise_conv2d_native(x, w_avg, [1,1,1,1], padding='SAME')
#
#    # perform gaussian filtering
#    sigma = gauss_sigma()
#
#    ax = tf.range(-filter_size//2+1, filter_size//2+1, dtype=tf.float32)
#    xx, yy = tf.meshgrid(ax, ax)
#
#    w_gauss_value = tf.exp(-(xx**2 + yy**2) / (2.0*sigma**2))
#    w_gauss_value = w_gauss_value / tf.reduce_sum(w_gauss_value)
#
#    w_gauss_value = tf.tile(tf.expand_dims(w_gauss_value, -1), [1,1, num_x])
#    w_gauss_value = tf.expand_dims(w_gauss_value,-1)
#    w_gauss = tf.Variable(w_gauss_value, name='w_gauss')
#    o_gauss = tf.nn.depthwise_conv2d_native(x, w_gauss, [1,1,1,1], padding='SAME')
#  
#
#    with tf.variable_scope(name) as scope:
#        # perform SSC filtering
#        fix_w = tf.get_variable('fix_w', shape=[fix_w_size, fix_w_size, 1, 1, 1], initializer=tf.zeros_initializer)
#        mask = np.zeros([fix_w_size, fix_w_size, 1, 1, 1], dtype=np.float32)
#        mask[dilation_factor - 1, dilation_factor - 1, 0, 0, 0] = 1
#        fix_w = tf.add(fix_w, tf.constant(mask, dtype=tf.float32))
#        o_ssc = tf.expand_dims(x, -1)
#        o_ssc = tf.nn.conv3d(o_ssc, fix_w, strides=[1,1,1,1,1], padding='SAME')
#        o_ssc = tf.squeeze(o_ssc, -1)
#
#        o = c_vector(o_avg, o_gauss, o_ssc)
#        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
#        o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
#        if biased:
#            b = tf.get_variable('biases', shape=[num_o])
#            o = tf.nn.bias_add(o, b)
#    return o

#def _get_c_vector(name):
#    with tf.variable_scope(name) as scope:
#        try:
##            c_ = tf.get_variable('c_vector', shape=[], initializer=tf.constant_initializer([0.33, 0.33, 0.33]))
#            c_values = [1.0, 1.0, 1.0]
#            c_init = tf.constant_initializer(c_values)
##            c_ = tf.get_variable('c_vector', shape=[3], initializer=c_init, constraint=lambda x: x / tf.reduce_sum(x))
##            c_ = tf.get_variable('c_vector', shape=[3], initializer=c_init )
#
#        except ValueError:
###            scope.reuse_variables()
#            c_ = tf.get_variable('c_vector')
#
#        return tf.nn.softmax(c_)

def _get_c_vector(name):
    with tf.variable_scope(name) as scope:
        try:
            c_value = [1.0]
            c_init = tf.constant_initializer(c_value)
#            c_1 = tf.get_variable('c_1', shape=[1], initializer=c_init )
#            c_2 = tf.get_variable('c_2', shape=[1], initializer=c_init )
#            c_3 = tf.get_variable('c_3', shape=[1], initializer=c_init )
            
            c_ = tf.get_variable('c_vector', shape=[3], initializer=tf.constant_initializer([0.33, 0.33, 0.33]))

        except ValueError:
            scope.reuse_variables()
#            c_1 = tf.get_variable('c_1')
#            c_2 = tf.get_variable('c_2')
#            c_3 = tf.get_variable('c_3')
            c_ = tf.get_variable('c_vector')

#        c_sum = tf.Variable
#
#        c_1.assign(c_1 / (c_1+c_2+c_3))
#        c_2.assign(c_2 / (c_1+c_2+c_3))
#        c_3.assign(c_3 / (c_1+c_2+c_3))


#        return c_1, c_2, c_3 

#        c_.assign(c_ / tf.reduce_sum(c_))
        c_.assign(tf.nn.softmax(c_)) 

        return c_


# Use a separate sigma value for each dilated convolution
def _combinational_layer(x, kernel_size, num_o, dilation_factor, name, top_scope, biased=False):
    """
    Dilated conv2d with antecedent average filter and without BN or relu.
    """
    num_x = x.shape[3].value
    filter_size = dilation_factor - 1
    fix_w_size = dilation_factor * 2 - 1

#    c_vector = tf.make_template('const_vector', _c_vector)

#    with tf.variable_scope(top_scope):
#        c_ = tf.get_variable('c_vector', shape=[3], initializer=tf.constant_initializer([0.33, 0.33, 0.33]))

#    c_1, c_2, c_3 = _get_c_vector(top_scope)
#    c_sum = tf.reduce_sum(c_)
#    c_.assign(c_ / c_sum)

    c_ = _get_c_vector(top_scope)

    
#    c_ = tf.nn.softmax(c_)
#    c_div = tf.div(c_, (c_[0]+c_[1]+c_[2]))
#    c_.assign(c_div) 




    # perform averaging (as seprable convolution)
    w_avg_value = 1.0/(filter_size*filter_size)
    w_avg = tf.Variable(tf.constant(w_avg_value,
                                    shape=[filter_size,filter_size,num_x,1]), name='w_avg',trainable=False)
    o_avg = tf.nn.depthwise_conv2d_native(x, w_avg, [1,1,1,1], padding='SAME')


    sigma = 1.0
    ax = tf.range(-filter_size//2+1, filter_size//2+1, dtype=tf.float32)
    xx, yy = tf.meshgrid(ax, ax)
    w_gauss_value = tf.exp(-(xx**2 + yy**2) / (2.0*sigma**2))
    w_gauss_value = w_gauss_value / tf.reduce_sum(w_gauss_value)
    w_gauss_value = tf.tile(tf.expand_dims(w_gauss_value, -1), [1,1, num_x])
    w_gauss_value = tf.expand_dims(w_gauss_value,-1)
    w_gauss = tf.Variable(w_gauss_value, name='w_gauss')
    o_gauss = tf.nn.depthwise_conv2d_native(x, w_gauss, [1,1,1,1], padding='SAME')

    t_ = [1.0, 0.0, 0.0]
#    t_ = [0.5, 0.5]

    o_t = t_[0]*o_avg + t_[1]*o_gauss
    with tf.variable_scope(name) as scope:
        # perform gaussian filtering
#        sigma_init = 1.0
#        init = tf.constant_initializer(sigma_init)
#        sigma = tf.get_variable('gauss_sigma', shape=[1], initializer=init)

        # perform SSC filtering
        fix_w = tf.get_variable('fix_w', shape=[fix_w_size, fix_w_size, 1, 1, 1], initializer=tf.zeros_initializer)
        mask = np.zeros([fix_w_size, fix_w_size, 1, 1, 1], dtype=np.float32)
        mask[dilation_factor - 1, dilation_factor - 1, 0, 0, 0] = 1
        fix_w = tf.add(fix_w, tf.constant(mask, dtype=tf.float32))
        o_ssc = tf.expand_dims(x, -1)
        o_ssc = tf.nn.conv3d(o_ssc, fix_w, strides=[1,1,1,1,1], padding='SAME')
        o_ssc = tf.squeeze(o_ssc, -1)

#        o = c_vector(o_avg, o_gauss, o_ssc)

#        o = c_[0]*o_avg + c_[1]*o_gauss + c_[2]*o_ssc



        o = t_[0]*o_avg + t_[1]*o_gauss + t_[2]*o_ssc
        o = o_avg

#        c_soft = tf.get_variable('c_soft_max', shape=[3], initializer=tf.zeros_initializer)
#        c_soft.assign(tf.nn.softmax(c_))
#        o = c_soft[0]*o_avg + c_soft[1]*o_gauss + c_soft[2]*o_ssc

#        o = c_1*o_avg + c_2*o_gauss + c_3*o_ssc
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
#        o = tf.nn.atrous_conv2d(o_t, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o



# Use a separate sigma for each dilated convolution 
def _gaussian_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope=1, biased=False):
    """
    Dilated conv2d with antecedent gaussian filter and without BN or relu.
    """
    num_x = x.shape[3].value
    filter_size = dilation_factor-1

    with tf.variable_scope(name) as scope:
        sigma_init = 1.0
        init = tf.constant_initializer(sigma_init)
        sigma = tf.get_variable('gauss_sigma', shape=[1], initializer=init)

#        ax = tf.range(-filter_size//2+1, filter_size//2+1, dtype=tf.float32)
#        xx, yy = tf.meshgrid(ax, ax)

#        mask_g = tf.exp(-(xx**2 + yy**2) / (2.0*sigma**2))
#        mask_g = mask_g / tf.reduce_sum(mask_g)


        ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-(xx**2 + yy**2)) 
        w_gauss_value = tf.Variable(tf.constant(0.0,
                                    shape=[filter_size,filter_size, 1,1,1]), name='w_gauss_value',trainable=False)

        mask = np.zeros([filter_size,filter_size, 1, 1, 1], dtype=np.float32)
        mask[:, :, 0, 0, 0] = kernel

        w_gauss_value = tf.add(w_gauss_value, tf.constant(mask, dtype=tf.float32))

        w_gauss_value = tf.div(w_gauss_value, tf.exp(2.0 * sigma**2))
        w_gauss_value = tf.div(w_gauss_value, tf.reduce_sum(w_gauss_value))


        o = tf.expand_dims(x, -1)
        o = tf.nn.conv3d(o, w_gauss_value, strides=[1,1,1,1,1], padding='SAME')
        o = tf.squeeze(o, -1)
 

#        w_gauss_value = tf.tile(tf.expand_dims(w_gauss_value, -1), [1,1, num_x])
#        w_gauss_value = tf.expand_dims(w_gauss_value,-1)
#        w_gauss = tf.Variable(w_gauss_value, name='w_gauss')

#        o = tf.nn.depthwise_conv2d_native(x, w_gauss, [1,1,1,1], padding='SAME')
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o



def _get_sigma(name):
    with tf.variable_scope(name) as scope:
        try:
#            c_ = tf.get_variable('c_vector', shape=[], initializer=tf.constant_initializer([0.33, 0.33, 0.33]))
            sigma_init = 1.0
            init = tf.constant_initializer(sigma_init)
            sigma = tf.get_variable('gauss_sigma', shape=[1], initializer=init)

        except ValueError:
            scope.reuse_variables()
            sigma_ = tf.get_variable('gauss_sigma')

        return sigma 




## Use a single sigma for all dilated convolutions
#def _gaussian_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope=1, biased=False):
#    """
#    Dilated conv2d with antecedent gaussian filter and without BN or relu.
#    """
#    num_x = x.shape[3].value
#    filter_size = dilation_factor-1
#
##    gauss_sigma = tf.make_template('sigma', _gauss_sigma)
##    sigma = gauss_sigma()
#    sigma = _get_sigma(name)
#
#    ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
#    xx, yy = np.meshgrid(ax, ax)
#
#    kernel = np.exp(-(xx**2 + yy**2)) 
#    w_gauss_value = tf.Variable(tf.constant(0.0,
#                                shape=[filter_size,filter_size, 1,1,1]), name='w_gauss_value',trainable=False)
#
#    mask = np.zeros([filter_size,filter_size, 1, 1, 1], dtype=np.float32)
#    mask[:, :, 0, 0, 0] = kernel
#
#    w_gauss_value = tf.add(w_gauss_value, tf.constant(mask, dtype=tf.float32))
#
#    w_gauss_value = tf.div(w_gauss_value, tf.exp(2.0 * sigma**2))
#    w_gauss_value = tf.div(w_gauss_value, tf.reduce_sum(w_gauss_value))
#
#
#    o = tf.expand_dims(x, -1)
#    o = tf.nn.conv3d(o, w_gauss_value, strides=[1,1,1,1,1], padding='SAME')
#    o = tf.squeeze(o, -1)
#
##    ax = tf.range(-filter_size//2+1, filter_size//2+1, dtype=tf.float32)
##    xx, yy = tf.meshgrid(ax, ax)
##
##    w_gauss_value = tf.exp(-(xx**2 + yy**2) / (2.0*sigma**2))
##    w_gauss_value = w_gauss_value / tf.reduce_sum(w_gauss_value)
##
##    w_gauss_value = tf.tile(tf.expand_dims(w_gauss_value, -1), [1,1, num_x])
##    w_gauss_value = tf.expand_dims(w_gauss_value,-1)
##    w_gauss = tf.Variable(w_gauss_value, name='w_gauss')
##    o_gauss = tf.nn.depthwise_conv2d_native(x, w_gauss, [1,1,1,1], padding='SAME')
#
#    with tf.variable_scope(name) as scope:
#        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
#        o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
#        if biased:
#            b = tf.get_variable('biases', shape=[num_o])
#            o = tf.nn.bias_add(o, b)
#
#        return o

def _decomposed_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope=1, biased=False):
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

def _smoothed_dilated_conv2d_GI(x, kernel_size, num_o, dilation_factor, name, top_scope=1, biased=False):
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

def _smoothed_dilated_conv2d_SSC(x, kernel_size, num_o, dilation_factor, name, top_scope=1, biased=False):
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
