from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import json
import os
import glob
import sys

class BaseNN:

    def __init__(self):
        self.data_description = {}
        self.keys_to_features = {}
        self.json_filename = ""
        self.data_description = {}
        self.keys_to_features = {}
        self.num_channels = 1
        self.out_channels = 1
        self.global_step = 0

    def set_global_step(self, step):
        self.global_step = step

    def get_global_step(self):
        return self.global_step

    def set_data_description(self, json_filename=None, data_description=None):

        if(json_filename):
            with open(json_filename, "r") as f:
                self.json_filename = json_filename
                self.data_description = json.load(f)
        elif(data_description):
            self.data_description = data_description

        if("data_keys" in self.data_description):
            for data_key in self.data_description["data_keys"]:
                self.keys_to_features[data_key] = tf.FixedLenFeature((np.prod(self.data_description[data_key]["shape"])), eval(self.data_description[data_key]["type"]))
        else:
            print("Nothing to decode! data_keys missing in object description object. tfRecords.py creates this descriptor.")
            raise
    def get_data_description(self):
        return self.data_description
    
    def read_and_decode(self, record):

        parsed = tf.parse_single_example(record, self.keys_to_features)
        reshaped_parsed = []

        if("data_keys" in self.data_description):
            for data_key in self.data_description["data_keys"]:
                reshaped_parsed.append(tf.reshape(parsed[data_key], self.data_description[data_key]["shape"]))

            return tuple(reshaped_parsed)

        return parsed

    def inputs(self, batch_size=1, num_epochs=1, buffer_size=1000):

        tfrecords_arr = []
        tfrecords_dir = os.path.join(os.path.dirname(self.json_filename), self.data_description["tfrecords"], '**/*.tfrecord')
        for tfr in glob.iglob(tfrecords_dir, recursive=True):
          tfrecords_arr.append(tfr)

        if((1, 15, 0) <= tuple([int(num) for num in tf.__version__.split('.')])):
            tf.data.experimental.ignore_errors()

        dataset = tf.data.TFRecordDataset(tfrecords_arr)
        dataset = dataset.map(self.read_and_decode)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_initializable_iterator()

        return iterator

    def print_tensor_shape(self, tensor, string):

    # input: tensor and string to describe it

        if __debug__:
            print('DEBUG ' + string, tensor.get_shape())
    # x=[batch, in_height, in_width, in_channels]
    # filter_shape=[filter_height, filter_width, in_channels, out_channels]
    def convolution2d(self, x, filter_shape, name='conv2d', strides=[1,1,1,1], activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

        with tf.variable_scope(name):
            with tf.device(ps_device):
                w_conv_name = 'w_' + name
                # filter_shape=[in_time,in_channels,out_channels]
                w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                self.print_tensor_shape( w_conv, name + ' weight shape')

                b_conv_name = 'b_' + name
                b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-1]])
                self.print_tensor_shape( b_conv, name + ' bias shape')

            with tf.device(w_device):
                conv_op = tf.nn.conv2d( x, w_conv, strides=strides, padding=padding, name='conv_op' )
                self.print_tensor_shape( conv_op, name + ' shape')

                conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

                if(activation):
                    conv_op = activation( conv_op, name='relu_op' ) 
                    self.print_tensor_shape( conv_op, name + ' relu_op shape')

                return conv_op

    # x=[batch, height, width, in_channels]
    # filter_shape=[height, width, output_channels, in_channels]
    def up_convolution2d(self, x, filter_shape, output_shape, name='up_conv', strides=[1,1,1,1], activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

        with tf.variable_scope(name):
            with tf.device(ps_device):
                w_conv_name = 'w_' + name
                w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                self.print_tensor_shape( w_conv, name + ' weight shape')

                b_conv_name = 'b_' + name
                b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-2]])
                self.print_tensor_shape( b_conv, name + ' bias shape')

            with tf.device(w_device):
                conv_op = tf.nn.conv2d_transpose( x, w_conv, output_shape=output_shape, strides=strides, padding=padding, name='up_conv_op' )
                self.print_tensor_shape( conv_op, name + ' shape')

                conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

                if(activation):
                    conv_op = activation( conv_op, name='relu_op' ) 
                    self.print_tensor_shape( conv_op, name + ' relu_op shape')

                return conv_op

    # x=[batch, in_depth, in_height, in_width, in_channels]
    # filter_shape=[filter_depth, filter_height, filter_width, in_channels, out_channels]
    def convolution3d(self, x, filter_shape, name='conv3d', strides=[1,1,1,1,1], activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

        with tf.variable_scope(name):
            with tf.device(ps_device):
                w_conv_name = 'w_' + name
                # filter_shape=[in_time,in_channels,out_channels]
                w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                self.print_tensor_shape( w_conv, name + ' weight shape')

                b_conv_name = 'b_' + name
                b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-1]])
                self.print_tensor_shape( b_conv, name + ' bias shape')

            with tf.device(w_device):
                conv_op = tf.nn.conv3d( x, w_conv, strides=strides, padding=padding, name='conv_op' )
                self.print_tensor_shape( conv_op, name + ' shape')

                conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

                if(activation):
                    conv_op = activation( conv_op, name='relu_op' ) 
                    self.print_tensor_shape( conv_op, name + ' relu_op shape')

                return conv_op

    # x=[batch, in_depth, in_height, in_width, in_channels]
    # filter_shape=[filter_depth, filter_height, filter_width, in_channels, out_channels]
    def up_convolution3d(self, x, filter_shape, output_shape, name='up_conv3d', strides=[1,1,1,1,1], activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

        with tf.variable_scope(name):
            with tf.device(ps_device):
                w_conv_name = 'w_' + name
                w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                self.print_tensor_shape( w_conv, name + ' weight shape')

                b_conv_name = 'b_' + name
                b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-2]])
                self.print_tensor_shape( b_conv, name + ' bias shape')

            with tf.device(w_device):
                conv_op = tf.nn.conv3d_transpose( x, w_conv, output_shape=output_shape, strides=strides, padding=padding, name='up_conv_op' )
                self.print_tensor_shape( conv_op, name + ' shape')

                conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

                if(activation):
                    conv_op = activation( conv_op, name='relu_op' ) 
                    self.print_tensor_shape( conv_op, name + ' relu_op shape')

                return conv_op

    # x=[batch, in_height, in_width, in_channels]
    # kernel=[k_batch, k_height, k_width, k_channels]
    def max_pool3d(self, x, name='max_pool', kernel=[1,3,3,3,1], strides=[1,2,2,2,1], padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

        with tf.variable_scope(name):
            with tf.device(w_device):
                pool_op = tf.nn.max_pool3d( x, kernel, strides=strides, padding=padding, name='max_pool_op' )
                self.print_tensor_shape( pool_op, name + ' shape')

                return pool_op

    # x=[batch, in_height, in_width, in_channels]
    # kernel=[k_batch, k_height, k_width, k_channels]
    def max_pool(self, x, name='max_pool', kernel=[1,3,3,1], strides=[1,2,2,1], padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

        with tf.variable_scope(name):
            with tf.device(w_device):
                pool_op = tf.nn.max_pool( x, kernel, strides=strides, padding=padding, name='max_pool_op' )
                self.print_tensor_shape( pool_op, name + ' shape')

                return pool_op

    def avg_pool3d(self, x, name='avg_pool', kernel=[1,3,3,3,1], strides=[1,2,2,2,1], padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

        with tf.variable_scope(name):
            with tf.device(w_device):
                pool_op = tf.nn.avg_pool3d( x, kernel, strides=strides, padding=padding, name='max_pool_op' )
                self.print_tensor_shape( pool_op, name + ' shape')

                return pool_op

    def avg_pool(self, x, name='avg_pool', kernel=[1,3,3,1], strides=[1,2,2,1], padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

        with tf.variable_scope(name):
            with tf.device(w_device):
                pool_op = tf.nn.avg_pool( x, kernel, strides=strides, padding=padding, name='max_pool_op' )
                self.print_tensor_shape( pool_op, name + ' shape')

                return pool_op

    def convolution(self, x, filter_shape, name='conv', stride=1, activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

        with tf.variable_scope(name):
    # weight variable 4d tensor, first two dims are patch (kernel) size       
    # third dim is number of input channels and fourth dim is output channels
            with tf.device(ps_device):

                w_conv_name = 'w_' + name
                # in_time -> stride in time
                # filter_shape=[in_time,in_channels,out_channels]
                w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
                self.print_tensor_shape( w_conv, name + ' weight shape')

                b_conv_name = 'b_' + name
                b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-1]])
                self.print_tensor_shape( b_conv, name + ' bias shape')

            with tf.device(w_device):
                conv_op = tf.nn.conv1d( x, w_conv, stride=stride, padding=padding, name='conv1_op' )
                self.print_tensor_shape( conv_op, name + ' shape')

                conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

                if(activation):
                    conv_op = activation( conv_op, name='relu_op' ) 
                    self.print_tensor_shape( conv_op, name + ' relu_op shape')

                return conv_op

    def matmul(self, x, out_channels, name='matmul', activation=tf.nn.relu, ps_device="/cpu:0", w_device="/gpu:0"):

        with tf.variable_scope(name):

            in_channels = x.get_shape().as_list()[-1]

        with tf.device(ps_device):
            w_matmul_name = 'w_' + name
            w_matmul = tf.get_variable(w_matmul_name, shape=[in_channels,out_channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))

            self.print_tensor_shape( w_matmul, name + ' shape')

            b_matmul_name = 'b_' + name
            b_matmul = tf.get_variable(name=b_matmul_name, shape=[out_channels])        

        with tf.device(w_device):

            matmul_op = tf.nn.bias_add(tf.matmul(x, w_matmul), b_matmul)

            if(activation):
                matmul_op = activation(matmul_op)

            return matmul_op
        
    def inference(self, data_tuple=None, images=None, keep_prob=1, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):
        raise NotImplementedError
    def metrics(self, logits, data_tuple, name='collection_metrics'):
        raise NotImplementedError
    def training(self, loss, learning_rate=1e-3, decay_steps=10000, decay_rate=0.96, staircase=False):
        raise NotImplementedError
    def loss(self, logits, data_tuple, class_weights=None):
        raise NotImplementedError
    def predict(self, logits):
        return logits
    def prediction_type(self):
        print("Implement in your own class, the return type must be one of 'image', 'segmentation', 'class', 'scalar'", file=sys.stderr)
        raise NotImplementedError