from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import json
import os
import glob
from . import base_nn
import sys

class NN(base_nn.BaseNN):

    def set_data_description(self, json_filename=None, data_description=None):
        super(NN, self).set_data_description(json_filename=json_filename, data_description=data_description)

        self.num_scalars = 1

    def inference(self, data_tuple=None, images=None, keep_prob=1, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

        if(data_tuple):
            images = data_tuple[0]

    #   input: tensor of images
    #   output: tensor of computed logits
        self.print_tensor_shape(images, "images")

        shape = tf.shape(images)
        batch_size = shape[0]
        num_channels = images.get_shape().as_list()[-1]

        images = tf.layers.batch_normalization(images, training=is_training)

        conv1_0 = self.convolution2d(images, name="conv1_0_op", filter_shape=[5, 5, num_channels, 16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        # conv1_1 = self.convolution2d(conv1_0, name="conv1_1_op", filter_shape=[5, 5, 16, 16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        pool1_0 = self.max_pool(conv1_0, name="pool1_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

        conv2_0 = self.convolution2d(pool1_0, name="conv2_0_op", filter_shape=[5, 5, 16, 32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        # conv2_1 = self.convolution2d(conv2_0, name="conv2_1_op", filter_shape=[5, 5, 32, 32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        pool2_0 = self.max_pool(conv2_0, name="pool2_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

        conv3_0 = self.convolution2d(pool2_0, name="conv3_0_op", filter_shape=[5, 5, 32, 64], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        # conv3_1 = self.convolution2d(conv3_0, name="conv3_1_op", filter_shape=[5, 5, 64, 64], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        pool3_0 = self.max_pool(conv3_0, name="pool3_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

        conv4_0 = self.convolution2d(pool3_0, name="conv4_0_op", filter_shape=[5, 5, 64, 128], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        # conv4_1 = self.convolution2d(conv4_0, name="conv4_1_op", filter_shape=[5, 5, 128, 128], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        pool4_0 = self.max_pool(conv4_0, name="pool4_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
        
        conv5_0 = self.convolution2d(pool4_0, name="conv5_0_op", filter_shape=[5, 5, 128, 256], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        #conv5_1 = self.convolution2d(conv5_0, name="conv5_1_op", filter_shape=[5, 5, 256, 256], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        pool5_0 = self.max_pool(conv5_0, name="pool5_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

        conv6_0 = self.convolution2d(pool5_0, name="conv6_0_op", filter_shape=[5, 5, 256, 512], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        #conv6_1 = self.convolution2d(conv6_0, name="conv6_1_op", filter_shape=[5, 5, 340, 340], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        pool6_0 = self.max_pool(conv6_0, name="pool6_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

        conv7_0 = self.convolution2d(pool6_0, name="conv7_0_op", filter_shape=[5, 5, 512, 1024], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        #conv6_1 = self.convolution2d(conv6_0, name="conv6_1_op", filter_shape=[5, 5, 340, 340], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        pool7_0 = self.max_pool(conv7_0, name="pool7_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

        
        kernel_size = pool7_0.get_shape().as_list()
        kernel_size[0] = 1
        kernel_size[-1] = 1
        pool8_0_op = self.max_pool(pool7_0, name="pool8_0_op", kernel=kernel_size, strides=kernel_size, ps_device=ps_device, w_device=w_device)        
        pool8_0_op = tf.reshape(pool8_0_op, (batch_size, 1024))
        
        # conv6_0_flat = tf.reshape(pool6_0, (batch_size, 27200))
        # matmul_7_0 = self.matmul(conv6_0_flat, 4096, name='matmul_7_0_op', activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        matmul_9_0 = self.matmul(pool8_0_op, 1024, name='matmul_9_0_op', activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        matmul_9_1 = self.matmul(matmul_9_0, 128, name='matmul_9_1_op', activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)

        final = self.matmul(matmul_9_1, self.num_scalars, name='final_op', activation=None, ps_device=ps_device, w_device=w_device)

        return final

    def metrics(self, logits, data_tuple, name='collection_metrics'):

        labels = data_tuple[1]
        with tf.variable_scope(name):
            weight_map = None

            self.print_tensor_shape( logits, 'logits metrics shape')
            self.print_tensor_shape( labels, 'labels metrics shape')
            
            metrics_obj = {}
            metrics_obj["MEAN_ABSOLUTE_ERROR"] = tf.metrics.mean_absolute_error(predictions=logits, labels=labels, weights=weight_map, name='mean_absolute_error')
            metrics_obj["MEAN_SQUARED_ERROR"] = tf.metrics.mean_squared_error(predictions=logits, labels=labels, weights=weight_map, name='mean_squared_error')
            # metrics_obj["AUC"] = tf.metrics.auc(predictions=logits_auc, labels=labels_auc, weights=weight_map, name='auc')
            # metrics_obj["FN"] = tf.metrics.false_negatives(predictions=logits, labels=labels, weights=weight_map, name='false_negatives')
            # metrics_obj["FP"] = tf.metrics.false_positives(predictions=logits, labels=labels, weights=weight_map, name='false_positives')
            # metrics_obj["TN"] = tf.metrics.true_negatives(predictions=logits, labels=labels, weights=weight_map, name='true_negatives')
            # metrics_obj["TP"] = tf.metrics.true_positives(predictions=logits, labels=labels, weights=weight_map, name='true_positives')
            
            for key in metrics_obj:
                tf.summary.scalar(key, metrics_obj[key][0])

            return metrics_obj


    def training(self, loss, learning_rate, decay_steps, decay_rate):
        
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

    def loss(self, logits, data_tuple, class_weights=None):
        
        labels = data_tuple[1]

        self.print_tensor_shape( logits, 'logits shape')
        self.print_tensor_shape( labels, 'labels shape')

        return tf.losses.absolute_difference(predictions=logits, labels=labels)

    def prediction_type(self):
        return "scalar"