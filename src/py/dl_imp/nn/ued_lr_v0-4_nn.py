from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import json
import os
import glob
from . import base_nn

class NN(base_nn.BaseNN):

    def set_data_description(self, json_filename=None, data_description=None):
        super(NN, self).set_data_description(json_filename=json_filename, data_description=data_description)
        self.num_channels = 1
        self.out_channels = 1

        if "data_keys" in self.data_description:
            data_keys = self.data_description["data_keys"]
            if data_keys[0] in self.data_description and "shape" in self.data_description[data_keys[0]]:
                self.num_channels = self.data_description[data_keys[0]]["shape"][-1]

            if data_keys[1] in self.data_description and "shape" in self.data_description[data_keys[1]]:
                self.out_channels = self.data_description[data_keys[1]]["shape"][-1]


    def inference(self, data_tuple=None, images=None, keep_prob=1, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

        if(data_tuple):
            images = data_tuple[0]

    #   input: tensor of images
    #   output: tensor of computed logits
        self.print_tensor_shape(images, "images")

        shape = tf.shape(images)
        batch_size = shape[0]

        with tf.variable_scope("layer1"):
            images = tf.layers.batch_normalization(images, training=is_training)

            conv1_0 = self.convolution2d(images, name="conv1_0_op", filter_shape=[3,3,self.num_channels,8], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
            conv1_1 = self.convolution2d(conv1_0, name="conv1_1_op", filter_shape=[3,3,8,8], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
            pool1_0 = self.max_pool(conv1_1, name="pool1_0_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

        conv2_0 = self.convolution2d(pool1_0, name="conv2_0_op", filter_shape=[3,3,8,16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        conv2_1 = self.convolution2d(conv2_0, name="conv2_1_op", filter_shape=[3,3,16,16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        pool2_0 = self.max_pool(conv2_1, name="pool2_0_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

        conv3_0 = self.convolution2d(pool2_0, name="conv3_0_op", filter_shape=[3,3,16,32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        conv3_1 = self.convolution2d(conv3_0, name="conv3_1_op", filter_shape=[3,3,32,32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        pool3_0 = self.max_pool(conv3_1, name="pool3_0_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

        conv4_0 = self.convolution2d(pool3_0, name="conv4_0_op", filter_shape=[3,3,32,64], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        conv4_1 = self.convolution2d(conv4_0, name="conv4_1_op", filter_shape=[3,3,64,64], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        up_out_shape_4 = conv3_1.get_shape().as_list()
        up_out_shape_4[0] = batch_size
        up_conv4_0 = self.up_convolution2d(conv4_1, name="up_conv4_0_op", filter_shape=[3,3,32,64], output_shape=up_out_shape_4, strides=[1,2,2,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        up_conv4_0 = self.max_pool(up_conv4_0, name="pool4_0_op", kernel=[1,3,3,1], strides=[1,1,1,1], ps_device=ps_device, w_device=w_device)
        
        concat5_0 = tf.concat([up_conv4_0, tf.nn.dropout( conv3_1, keep_prob)], -1)

        conv5_0 = self.convolution2d(concat5_0, name="conv5_0_op", filter_shape=[3,3,64,32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        conv5_1 = self.convolution2d(conv5_0, name="conv5_1_op", filter_shape=[3,3,32,32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        up_out_shape_5 = conv2_1.get_shape().as_list()
        up_out_shape_5[0] = batch_size
        up_conv5_0 = self.up_convolution2d(conv5_1, name="up_conv5_0_op", filter_shape=[3,3,16,32], output_shape=up_out_shape_5, strides=[1,2,2,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        up_conv5_0 = self.max_pool(up_conv5_0, name="pool5_0_op", kernel=[1,3,3,1], strides=[1,1,1,1], ps_device=ps_device, w_device=w_device)
        
        concat6_0 = tf.concat([up_conv5_0, tf.nn.dropout( conv2_1, keep_prob)], -1)

        conv6_0 = self.convolution2d(concat6_0, name="conv6_0_op", filter_shape=[3,3,32,16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        conv6_1 = self.convolution2d(conv6_0, name="conv6_1_op", filter_shape=[3,3,16,16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        up_out_shape_6 = conv1_1.get_shape().as_list()
        up_out_shape_6[0] = batch_size
        up_conv6_0 = self.up_convolution2d(conv6_1, name="up_conv6_0_op", filter_shape=[3,3,8,16], output_shape=up_out_shape_6, strides=[1,2,2,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        up_conv6_0 = self.max_pool(up_conv6_0, name="pool6_0_op", kernel=[1,3,3,1], strides=[1,1,1,1], ps_device=ps_device, w_device=w_device)
        
        concat7_0 = tf.concat([up_conv6_0, tf.nn.dropout( conv1_1, keep_prob)], -1)

        conv7_0 = self.convolution2d(concat7_0, name="conv7_0_op", filter_shape=[7,7,16,16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        conv7_1 = self.convolution2d(conv7_0, name="conv7_1_op", filter_shape=[7,7,16,16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)

        conv7_2 = self.convolution2d(conv7_1, name="conv7_2_op", filter_shape=[1,1,16,16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        final = self.convolution2d(conv7_2, name="final", filter_shape=[1,1,16,self.out_channels], strides=[1,1,1,1], padding="SAME", activation=None, ps_device=ps_device, w_device=w_device)

        return final

    def metrics(self, logits, data_tuple, name='collection_metrics'):

        labels = data_tuple[1]
        with tf.variable_scope(name):
            weight_map = None

            metrics_obj = {}
            
            metrics_obj["MEAN_ABSOLUTE_ERROR"] = tf.metrics.mean_absolute_error(predictions=logits, labels=labels, weights=weight_map, name='mean_absolute_error')
            metrics_obj["MEAN_SQUARED_ERROR"] = tf.metrics.mean_squared_error(predictions=logits, labels=labels, weights=weight_map, name='mean_squared_error')
            metrics_obj["ROOT_MEAN_SQUARED_ERROR"] = tf.metrics.root_mean_squared_error(predictions=logits, labels=labels, weights=weight_map, name='root_mean_squared_error')
            
            
            for key in metrics_obj:
                tf.summary.scalar(key, metrics_obj[key][0])

            return metrics_obj


    def training(self, loss, learning_rate, decay_steps, decay_rate, staircase):
        
        global_step = tf.Variable(self.global_step, name='global_step', trainable=False)

        # create learning_decay
        lr = tf.train.exponential_decay( learning_rate,
                                         global_step,
                                         decay_steps,
                                         decay_rate, staircase=staircase )

        tf.summary.scalar('2learning_rate', lr )

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(lr)

        # Use the optimizer to apply the gradients that minimize the loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def loss(self, logits, data_tuple, class_weights=None):
        
        labels = data_tuple[1]

        self.print_tensor_shape( logits, 'logits shape')
        self.print_tensor_shape( labels, 'labels shape')

        shape = tf.shape(logits)
        batch_size = shape[0]

        with tf.variable_scope("layer1", reuse=True):
            labels_conv = tf.layers.batch_normalization(labels, training=False)
            labels_conv = self.convolution2d(labels_conv, name="conv1_0_op", filter_shape=[3,3,self.num_channels,8], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device="/cpu:0", w_device="/gpu:0")
            labels_conv = self.convolution2d(labels_conv, name="conv1_1_op", filter_shape=[3,3,8,8], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device="/cpu:0", w_device="/gpu:0")
            labels_conv = self.max_pool(labels_conv, name="pool1_0_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device="/cpu:0", w_device="/gpu:0")

        with tf.variable_scope("layer1", reuse=True):
            logits_conv = tf.layers.batch_normalization(logits, training=False)
            logits_conv = self.convolution2d(logits_conv, name="conv1_0_op", filter_shape=[3,3,self.num_channels,8], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device="/cpu:0", w_device="/gpu:0")
            logits_conv = self.convolution2d(logits_conv, name="conv1_1_op", filter_shape=[3,3,8,8], strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device="/cpu:0", w_device="/gpu:0")
            logits_conv = self.max_pool(logits_conv, name="pool1_0_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device="/cpu:0", w_device="/gpu:0")


        logits_flat = tf.reshape(logits, [batch_size, -1])
        labels_flat = tf.reshape(labels, [batch_size, -1])

        logits_conv_flat = tf.reshape(logits_conv, [batch_size, -1])
        labels_conv_flat = tf.reshape(labels_conv, [batch_size, -1])

        # return tf.pow(tf.norm(logits_flat - labels_flat), 2)/tf.math.reduce_prod(tf.dtypes.cast(shape, tf.float32)[1:])
        return 2.0*tf.losses.absolute_difference(labels=labels_conv_flat, predictions=logits_conv_flat) + tf.losses.absolute_difference(labels=labels_flat, predictions=logits_flat)

    def prediction_type(self):
        return "image"
