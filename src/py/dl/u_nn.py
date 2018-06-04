from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class ReadAndDecode:

    image_shape = None
    label_shape = None

    def read_and_decode(self, record):

        keys_to_features = {
          "image": tf.FixedLenFeature((np.prod(self.image_shape)), tf.float32),
          "label": tf.FixedLenFeature((np.prod(self.label_shape)), tf.int64)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        
        image = parsed["image"]
        label = tf.cast(parsed["label"], tf.int32)

        image = tf.reshape(image, self.image_shape)
        label = tf.reshape(label, self.label_shape)

        return image, label


def inputs(filenames, image_shape=None, label_shape=None, num_labels=2, batch_size=1, num_epochs=1):

    r_a_d = ReadAndDecode()
    r_a_d.image_shape = image_shape
    r_a_d.label_shape = label_shape
    r_a_d.num_labels = num_labels

    dataset = tf.data.TFRecordDataset(filenames)
    
    dataset = dataset.map(r_a_d.read_and_decode)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_initializable_iterator()

    return iterator

def print_tensor_shape(tensor, string):

# input: tensor and string to describe it

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())
# x=[batch, in_height, in_width, in_channels]
# filter_shape=[filter_height, filter_width, in_channels, out_channels]
def convolution2d(x, filter_shape, name='conv2d', strides=[1,1,1,1], activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):
        with tf.device(ps_device):
            w_conv_name = 'w_' + name
            # filter_shape=[in_time,in_channels,out_channels]
            w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_conv, name + ' weight shape')

            b_conv_name = 'b_' + name
            b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-1]])
            print_tensor_shape( b_conv, name + ' bias shape')

        with tf.device(w_device):
            conv_op = tf.nn.conv2d( x, w_conv, strides=strides, padding=padding, name='conv_op' )
            print_tensor_shape( conv_op, name + ' shape')

            conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

            if(activation):
                conv_op = activation( conv_op, name='relu_op' ) 
                print_tensor_shape( conv_op, name + ' relu_op shape')

            return conv_op

# x=[batch, height, width, in_channels]
# filter_shape=[height, width, output_channels, in_channels]
def up_convolution2d(x, filter_shape, output_shape, name='up_conv', strides=[1,1,1,1], activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):
        with tf.device(ps_device):
            w_conv_name = 'w_' + name
            w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_conv, name + ' weight shape')

            b_conv_name = 'b_' + name
            b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-2]])
            print_tensor_shape( b_conv, name + ' bias shape')

        with tf.device(w_device):
            conv_op = tf.nn.conv2d_transpose( x, w_conv, output_shape=output_shape, strides=strides, padding=padding, name='up_conv_op' )
            print_tensor_shape( conv_op, name + ' shape')

            conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

            if(activation):
                conv_op = activation( conv_op, name='relu_op' ) 
                print_tensor_shape( conv_op, name + ' relu_op shape')

            return conv_op

# x=[batch, in_height, in_width, in_channels]
# kernel=[k_batch, k_height, k_width, k_channels]
def max_pool(x, name='max_pool', kernel=[1,3,3,1], strides=[1,2,2,1], padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):
        with tf.device(w_device):
            pool_op = tf.nn.max_pool( x, kernel, strides=strides, padding=padding, name='max_pool_op' )
            print_tensor_shape( pool_op, name + ' shape')

            return pool_op

def convolution(x, filter_shape, name='conv', stride=1, activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):
# weight variable 4d tensor, first two dims are patch (kernel) size       
# third dim is number of input channels and fourth dim is output channels
        with tf.device(ps_device):

            w_conv_name = 'w_' + name
            # in_time -> stride in time
            # filter_shape=[in_time,in_channels,out_channels]
            w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_conv, name + ' weight shape')

            b_conv_name = 'b_' + name
            b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-1]])
            print_tensor_shape( b_conv, name + ' bias shape')

        with tf.device(w_device):
            conv_op = tf.nn.conv1d( x, w_conv, stride=stride, padding=padding, name='conv1_op' )
            print_tensor_shape( conv_op, name + ' shape')

            conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

            if(activation):
                conv_op = activation( conv_op, name='relu_op' ) 
                print_tensor_shape( conv_op, name + ' relu_op shape')

            return conv_op

def matmul(x, out_channels, name='matmul', activation=tf.nn.relu, ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):

        in_channels = x.get_shape().as_list()[-1]

    with tf.device(ps_device):
        w_matmul_name = 'w_' + name
        w_matmul = tf.get_variable(w_matmul_name, shape=[in_channels,out_channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))

        print_tensor_shape( w_matmul, name + ' shape')        

        b_matmul_name = 'b_' + name
        b_matmul = tf.get_variable(name=b_matmul_name, shape=[out_channels])        

    with tf.device(w_device):

        matmul_op = tf.nn.bias_add(tf.matmul(x, w_matmul), b_matmul)

        if(activation):
            matmul_op = activation(matmul_op)

        return matmul_op

def inference(images, num_labels=2, keep_prob=1, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

#   input: tensor of images
#   output: tensor of computed logits
    print_tensor_shape(images, "images")

    shape = tf.shape(images)
    batch_size = shape[0]
    num_channels = images.get_shape().as_list()[-1]

    images = tf.layers.batch_normalization(images, training=is_training)

    conv1_0 = convolution2d(images, name="conv1_0_op", filter_shape=[5, 5, num_channels, 8], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    conv1_1 = convolution2d(conv1_0, name="conv1_1_op", filter_shape=[5, 5, 8, 8], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    pool1_0 = max_pool(conv1_1, name="pool1_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

    conv2_0 = convolution2d(pool1_0, name="conv2_0_op", filter_shape=[5, 5, 8, 16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    conv2_1 = convolution2d(conv2_0, name="conv2_1_op", filter_shape=[5, 5, 16, 16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    pool2_0 = max_pool(conv2_1, name="pool2_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

    conv3_0 = convolution2d(pool2_0, name="conv3_0_op", filter_shape=[5, 5, 16, 32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    conv3_1 = convolution2d(conv3_0, name="conv3_1_op", filter_shape=[5, 5, 32, 32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    pool3_0 = max_pool(conv3_1, name="pool3_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
    
    pool3_0 = tf.nn.dropout( pool3_0, keep_prob)

    conv4_0 = convolution2d(pool3_0, name="conv4_0_op", filter_shape=[5, 5, 32, 64], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    conv4_1 = convolution2d(conv4_0, name="conv4_1_op", filter_shape=[5, 5, 64, 64], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    up_out_shape_4 = conv3_1.get_shape().as_list()
    up_out_shape_4[0] = batch_size
    up_conv4_0 = up_convolution2d(conv4_1, name="up_conv4_0_op", filter_shape=[5, 5, 32, 64], output_shape=up_out_shape_4, strides=[1,2,2,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    
    concat5_0 = tf.concat([up_conv4_0, conv3_1], -1)

    conv5_0 = convolution2d(concat5_0, name="conv5_0_op", filter_shape=[5, 5, 64, 32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    conv5_1 = convolution2d(conv5_0, name="conv5_1_op", filter_shape=[5, 5, 32, 32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    up_out_shape_5 = conv2_1.get_shape().as_list()
    up_out_shape_5[0] = batch_size
    up_conv5_0 = up_convolution2d(conv5_1, name="up_conv5_0_op", filter_shape=[5, 5, 16, 32], output_shape=up_out_shape_5, strides=[1,2,2,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    
    concat6_0 = tf.concat([up_conv5_0, conv2_1], -1)

    conv6_0 = convolution2d(concat6_0, name="conv6_0_op", filter_shape=[5, 5, 32, 16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    conv6_1 = convolution2d(conv6_0, name="conv6_1_op", filter_shape=[5, 5, 16, 16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    up_out_shape_6 = conv1_1.get_shape().as_list()
    up_out_shape_6[0] = batch_size
    up_conv6_0 = up_convolution2d(conv6_1, name="up_conv6_0_op", filter_shape=[5, 5, 8, 16], output_shape=up_out_shape_6, strides=[1,2,2,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    
    concat7_0 = tf.concat([up_conv6_0, conv1_1], -1)

    conv7_0 = convolution2d(concat7_0, name="conv7_0_op", filter_shape=[5, 5, 16, 8], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
    conv7_1 = convolution2d(conv7_0, name="conv7_1_op", filter_shape=[5, 5, 8, 8], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)

    final = convolution2d(conv7_1, name="final", filter_shape=[1, 1, 8, 1], strides=[1,1,1,1], padding="SAME", activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)

    return final

def metrics(logits, labels, class_weights=None, metrics_collections=None):

    weight_map = None

    if(class_weights is not None):
        weight_map = tf.multiply(labels, class_weights)

    accuracy = tf.metrics.accuracy(predictions=logits, labels=labels, weights=weight_map, name='accuracy', metrics_collections=metrics_collections)
    auc_eval = tf.metrics.auc(predictions=logits, labels=labels, weights=weight_map, name='auc', metrics_collections=metrics_collections)
    fn_eval = tf.metrics.false_negatives(predictions=logits, labels=labels, weights=weight_map, name='false_negatives', metrics_collections=metrics_collections)
    fp_eval = tf.metrics.false_positives(predictions=logits, labels=labels, weights=weight_map, name='false_positives', metrics_collections=metrics_collections)
    tn_eval = tf.metrics.true_negatives(predictions=logits, labels=labels, weights=weight_map, name='true_negatives', metrics_collections=metrics_collections)
    tp_eval = tf.metrics.true_positives(predictions=logits, labels=labels, weights=weight_map, name='true_positives', metrics_collections=metrics_collections)
    return accuracy,auc_eval,fn_eval,fp_eval,tn_eval,tp_eval


def training(loss, learning_rate, decay_steps, decay_rate):
    
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # create learning_decay
    lr = tf.train.exponential_decay( learning_rate,
                                     global_step,
                                     decay_steps,
                                     decay_rate, staircase=True )

    tf.summary.scalar('2learning_rate', lr )

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(lr)

    # Use the optimizer to apply the gradients that minimize the loss
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def loss(logits, labels, class_weights=None):
    
    print_tensor_shape( logits, 'logits shape')
    print_tensor_shape( labels, 'labels shape')

    shape = tf.shape(logits)
    batch_size = shape[0]

    logits_flat = tf.reshape(logits, [batch_size, -1])
    labels_flat = tf.reshape(tf.cast(labels, tf.float32), [batch_size, -1])

    intersection = 2.0 * tf.reduce_sum(logits_flat * labels_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(logits_flat, axis=1) + tf.reduce_sum(labels_flat, axis=1) + 1e-7

    return 1.0 - tf.reduce_mean(intersection / denominator)