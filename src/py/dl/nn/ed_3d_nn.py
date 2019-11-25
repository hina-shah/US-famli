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
            if len(data_keys) > 0 and data_keys[0] in self.data_description and "shape" in self.data_description[data_keys[0]]:
                self.num_channels = self.data_description[data_keys[0]]["shape"][-1]

        self.out_channels = self.num_channels
        self.variables = {}
        self.losses = {}


    def inference(self, data_tuple=None, images=None, keep_prob=1, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

        if(data_tuple):
            images = data_tuple[0]

    #   input: tensor of images
    #   output: tensor of computed logits
        self.print_tensor_shape(images, "images")

        shape = tf.shape(images)
        batch_size = shape[0]

        if(is_training):

            with tf.variable_scope("layer1") as scope:
                conv1 = tf.layers.batch_normalization(images, training=is_training)
                conv1 = self.convolution3d(conv1, name="conv1", filter_shape=[5,5,5,self.num_channels,32], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.convolution3d(conv1, name="conv2", filter_shape=[5,5,5,32,32], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.max_pool3d(conv1, name="pool1", kernel=[1,3,3,3,1], strides=[1,2,2,2,1], padding="SAME", ps_device=ps_device, w_device=w_device)
                conv1 = self.convolution3d(conv1, name="conv3", filter_shape=[1,1,1,32,8], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)


                self.variables["layer1_down"] = conv1
                
                conv1 = tf.layers.batch_normalization(conv1, training=is_training)
                conv1 = tf.nn.dropout(conv1, keep_prob)
                up_shape = images.get_shape().as_list()
                up_shape[0] = batch_size
                up_shape[-1] = 32
                conv1 = self.convolution3d(conv1, name="up_conv1", filter_shape=[5,5,5,8,32], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.up_convolution3d(conv1, name="up_pool1", filter_shape=[5,5,5,32,32], output_shape=up_shape, strides=[1,2,2,2,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.convolution3d(conv1, name="up_conv2", filter_shape=[5,5,5,32,32], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                final = self.convolution3d(conv1, name="up_conv3", filter_shape=[1,1,1,32,self.out_channels], strides=[1,1,1,1,1], padding="SAME", activation=None, ps_device=ps_device, w_device=w_device)
                

            with tf.variable_scope("layer2") as scope:
                conv1 = tf.layers.batch_normalization(self.variables["layer1_down"], training=is_training)
                conv1 = self.convolution3d(conv1, name="conv1", filter_shape=[5,5,5,8,128], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.convolution3d(conv1, name="conv2", filter_shape=[5,5,5,128,128], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.max_pool3d(conv1, name="pool1", kernel=[1,3,3,3,1], strides=[1,2,2,2,1], padding="SAME", ps_device=ps_device, w_device=w_device)
                conv1 = self.convolution3d(conv1, name="conv3", filter_shape=[1,1,1,128,32], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)

                self.variables["layer2_down"] = conv1
                conv1 = tf.layers.batch_normalization(conv1, training=is_training)
                conv1 = tf.nn.dropout(conv1, keep_prob)
                up_shape = self.variables["layer1_down"].get_shape().as_list()
                up_shape[0] = batch_size
                up_shape[-1] = 128
                conv1 = self.convolution3d(conv1, name="up_conv1", filter_shape=[5,5,5,32,128], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.up_convolution3d(conv1, name="up_pool1", filter_shape=[5,5,5,128,128], output_shape=up_shape, strides=[1,2,2,2,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.convolution3d(conv1, name="up_conv2", filter_shape=[5,5,5,128,128], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                self.variables["layer2_up"] = self.convolution3d(conv1, name="up_conv3", filter_shape=[1,1,1,128,8], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)

            return final

        else:

            with tf.variable_scope("layer1") as scope:
                conv1 = tf.layers.batch_normalization(images, training=is_training)
                conv1 = self.convolution3d(conv1, name="conv1", filter_shape=[5,5,5,self.num_channels,32], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.convolution3d(conv1, name="conv2", filter_shape=[5,5,5,32,32], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.max_pool3d(conv1, name="pool1", kernel=[1,3,3,3,1], strides=[1,2,2,2,1], padding="SAME", ps_device=ps_device, w_device=w_device)
                conv1 = self.convolution3d(conv1, name="conv3", filter_shape=[1,1,1,32,8], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)

                self.variables["layer1_down"] = conv1

            with tf.variable_scope("layer2") as scope:
                conv1 = self.convolution3d(self.variables["layer1_down"], name="conv1", filter_shape=[5,5,5,8,128], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.convolution3d(conv1, name="conv2", filter_shape=[5,5,5,128,128], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.max_pool3d(conv1, name="pool1", kernel=[1,3,3,3,1], strides=[1,2,2,2,1], padding="SAME", ps_device=ps_device, w_device=w_device)
                conv1 = self.convolution3d(conv1, name="conv3", filter_shape=[1,1,1,128,32], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)

                self.variables["layer2_down"] = conv1

                # return self.variables["layer2_down"]
            
                up_shape = self.variables["layer1_down"].get_shape().as_list()
                up_shape[0] = batch_size
                up_shape[-1] = 128
                conv1 = self.convolution3d(self.variables["layer2_down"], name="up_conv1", filter_shape=[5,5,5,32,128], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                return conv1
                conv1 = self.up_convolution3d(conv1, name="up_pool1", filter_shape=[5,5,5,128,128], output_shape=up_shape, strides=[1,2,2,2,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.convolution3d(conv1, name="up_conv2", filter_shape=[5,5,5,128,128], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                self.variables["layer2_up"] = self.convolution3d(conv1, name="up_conv3", filter_shape=[1,1,1,128,8], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)

                return self.variables["layer2_up"]

            with tf.variable_scope("layer1", reuse=tf.AUTO_REUSE) as scope:
                up_shape = images.get_shape().as_list()
                up_shape[0] = batch_size
                up_shape[-1] = 32
                conv1 = self.convolution3d(self.variables["layer2_up"], name="up_conv1", filter_shape=[5,5,5,8,32], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.up_convolution3d(conv1, name="up_pool1", filter_shape=[5,5,5,32,32], output_shape=up_shape, strides=[1,2,2,2,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                conv1 = self.convolution3d(conv1, name="up_conv2", filter_shape=[5,5,5,32,32], strides=[1,1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
                final = self.convolution3d(conv1, name="up_conv3", filter_shape=[1,1,1,32,self.out_channels], strides=[1,1,1,1,1], padding="SAME", activation=None, ps_device=ps_device, w_device=w_device)

            return final


    def metrics(self, logits, data_tuple, name='collection_metrics'):

        labels = data_tuple[0]
        with tf.variable_scope(name):
            weight_map = None

            metrics_obj = {}
            
            metrics_obj["MEAN_ABSOLUTE_ERROR_L1"] = tf.metrics.mean_absolute_error(predictions=logits, labels=labels, weights=weight_map, name='mean_absolute_error_l1')
            metrics_obj["MEAN_ABSOLUTE_ERROR_L2"] = tf.metrics.mean_absolute_error(predictions=self.variables["layer2_up"], labels=self.variables["layer1_down"], weights=weight_map, name='mean_absolute_error_l2')
            # metrics_obj["MEAN_ABSOLUTE_ERROR_L3"] = tf.metrics.mean_absolute_error(predictions=self.variables["layer3_up"], labels=self.variables["layer2_down"], weights=weight_map, name='mean_absolute_error_l3')
            
            
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

        optimizer = tf.train.AdamOptimizer(lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op = optimizer.minimize(tf.math.add(loss, self.losses["layer2_loss"]), global_step=global_step)

        return train_op

        # vars_train = tf.trainable_variables()

        # with tf.variable_scope("layer1") as scope:
        #     optimizer = tf.train.AdamOptimizer(lr)
        #     # Use the optimizer to apply the gradients that minimize the loss
        #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #     with tf.control_dependencies(update_ops):
        #       train_op_layer1 = optimizer.minimize(loss, global_step=global_step, var_list=[var for var in vars_train if 'layer1' in var.name])

        # with tf.variable_scope("layer2") as scope:
        #     optimizer = tf.train.AdamOptimizer(lr)
        #     # Use the optimizer to apply the gradients that minimize the loss
        #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #     with tf.control_dependencies(update_ops):
        #         print([var for var in vars_train if 'layer2' in var.name])
        #         train_op_layer2 = optimizer.minimize(self.losses["layer2_loss"], global_step=global_step, var_list=[var for var in vars_train if 'layer2' in var.name])

        # # with tf.variable_scope("layer3") as scope:
        # #     optimizer = tf.train.AdamOptimizer(lr)
        # #     # Use the optimizer to apply the gradients that minimize the loss
        # #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # #     with tf.control_dependencies(update_ops):
        # #       train_op_layer3 = optimizer.minimize(self.losses["layer3_loss"], global_step=global_step, var_list=[var for var in vars_train if 'layer3' in var.name])

        # # return tf.group(train_op_layer1, train_op_layer2, train_op_layer3)
        # return tf.group(train_op_layer1, train_op_layer2)

    def loss(self, logits, data_tuple, class_weights=None):

        labels = data_tuple[0]

        self.print_tensor_shape( logits, 'logits shape')
        self.print_tensor_shape( labels, 'labels shape')

        shape = tf.shape(logits)
        batch_size = shape[0]

        self.losses["layer2_loss"] = tf.losses.absolute_difference(predictions=tf.reshape(self.variables["layer2_up"], [batch_size, -1]), labels=tf.reshape(self.variables["layer1_down"], [batch_size, -1]))
        # self.losses["layer3_loss"] = tf.losses.absolute_difference(predictions=tf.reshape(self.variables["layer3_up"], [batch_size, -1]), labels=tf.reshape(self.variables["layer2_down"], [batch_size, -1]))

        logits_flat = tf.reshape(logits, [batch_size, -1])
        labels_flat = tf.reshape(labels, [batch_size, -1])

        return tf.losses.absolute_difference(predictions=logits_flat, labels=labels_flat)

    def prediction_type(self):
        return "image"
