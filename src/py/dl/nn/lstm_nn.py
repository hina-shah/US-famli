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
        
        self.num_channels = 1
        self.out_channels = 1
        
        if "data_keys" in self.data_description:
            data_keys = self.data_description["data_keys"]
            if data_keys[0] in self.data_description and "shape" in self.data_description[data_keys[0]]:
                self.num_channels = self.data_description[data_keys[0]]["shape"][-1]

    def read_and_decode(self, record):

        parsed = tf.parse_single_example(record, self.keys_to_features)
        reshaped_parsed = []

        if("data_keys" in self.data_description):
            for data_key in self.data_description["data_keys"]:
                
                print('reshaping key: {}',format(data_key))
                reshaped_parsed.append(tf.reshape(parsed[data_key], self.data_description[data_key]["shape"]))
                r = reshaped_parsed[-1].eval(session=tf.Session())
                self.print_tensor_shape(reshaped_parsed[-1], data_key)
            return tuple(reshaped_parsed)

        return parsed

    def upblock(self, x0, out_filters, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

        shape = tf.shape(x0)
        batch_size = shape[0]

        output_shape = x0.get_shape().as_list()
        in_filters = output_shape[-1]
        output_shape = [batch_size,output_shape[1]*2,output_shape[2]*2,out_filters]

        x = self.up_convolution2d(x0, name="up_conv1", filter_shape=[5,5,out_filters,in_filters], output_shape=output_shape, strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, initializer=tf.random_normal_initializer(mean=0,stddev=0.01), ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=True)
        x = tf.nn.leaky_relu(x)
        # x = self.max_pool(x, name="pool1", kernel=[1,3,3,1], strides=[1,1,1,1], ps_device=ps_device, w_device=w_device)
        # x = self.convolution2d(x, name="conv1", filter_shape=[5,5,out_filters,out_filters], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, initializer=tf.random_normal_initializer(mean=0,stddev=0.01), ps_device=ps_device, w_device=w_device)
        # x = tf.layers.batch_normalization(x, training=is_training)
        # x = tf.nn.leaky_relu(x)
        
        return x

    def inference(self, data_tuple=None, images=None, keep_prob=1, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

        with tf.variable_scope("generator"):
            
            
            batch_size = 1

            if(is_training):
                if(data_tuple):
                    shape = tf.shape(data_tuple[0])
                    batch_size = shape[0]
                x = tf.random.normal([batch_size,128])
            elif(data_tuple):
                x = tf.reshape(data_tuple[0], [batch_size, 128])
            else:
                x = tf.reshape(images, [batch_size, 128])

            is_training = True
            self.print_tensor_shape(x, "input_x")

            with tf.variable_scope("block0"):
                x = self.matmul(x, 4*4*1024, name='project', activation=None, initializer=tf.random_normal_initializer(mean=0,stddev=0.01), ps_device=ps_device, w_device=w_device)
                x = tf.reshape(x, [batch_size, 4, 4, 1024])
                x = tf.layers.batch_normalization(x, training=True)
                x = tf.nn.leaky_relu(x)

            with tf.variable_scope("block1"):
                x = self.upblock(x, 512, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("block2"):
                x = self.upblock(x, 256, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("block3"):
                x = self.upblock(x, 128,is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = tf.nn.dropout(x, keep_prob)
            
            with tf.variable_scope("block4"):
                x = self.up_convolution2d(x, name="up_conv1", filter_shape=[5,5,1,128], output_shape=[batch_size,64,64,1], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, initializer=tf.random_normal_initializer(mean=0,stddev=0.01), ps_device=ps_device, w_device=w_device)
                x = tf.nn.tanh(x)

        return x

    def discriminator(self, images=None, data_tuple=None, num_labels=2, keep_prob=1, is_training=False, reuse=False, ps_device="/cpu:0", w_device="/gpu:0"):
    #   input: tensor of images
    #   output: tensor of computed logits
        if(data_tuple):
            images = data_tuple[0]
        self.print_tensor_shape(images, "images")

        shape = tf.shape(images)
        batch_size = shape[0]
        
        with tf.variable_scope("discriminator") as scope:
            if(reuse):
                scope.reuse_variables()

            x = tf.layers.batch_normalization(images, training=is_training)

            with tf.variable_scope("block0"):
                x = self.convolution2d(x, name="conv1", filter_shape=[5,5,self.num_channels,64], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)
                x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
                # x = self.convolution2d(x, name="conv2", filter_shape=[5,5,64,64], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)
                # x = tf.nn.dropout(x, keep_prob)
                
            with tf.variable_scope("block1"):
                x = self.convolution2d(x, name="conv1", filter_shape=[5,5,64,128], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)
                x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
                # x = self.convolution2d(x, name="conv2", filter_shape=[5,5,128,128], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)
                # x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("block2"):
                x = self.convolution2d(x, name="conv1", filter_shape=[5,5,128,256], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)
                x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
                # x = self.convolution2d(x, name="conv2", filter_shape=[5,5,256,256], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)
                # x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("block3"):
                x = self.convolution2d(x, name="conv1", filter_shape=[5,5,256,512], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)            
                x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
                # x = self.convolution2d(x, name="conv2", filter_shape=[5,5,512,512], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)
                # x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("fc"):
                kernel_size = x.get_shape().as_list()
                kernel_size[0] = 1
                kernel_size[-1] = 1
                x = self.avg_pool(x, name="avg_pool_op", kernel=kernel_size, strides=kernel_size, ps_device=ps_device, w_device=w_device)
                x = tf.reshape(x, (batch_size, 512))

                x = self.matmul(x, 2, name='final_op', use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)

            return x


    def metrics(self, logits, labels, name='collection_metrics'):

        
        with tf.variable_scope(name):
            weight_map = None

            metrics_obj = {}
            
            metrics_obj["ACCURACY"] = tf.metrics.accuracy(predictions=logits, labels=labels, weights=weight_map, name='accuracy')            
            
            for key in metrics_obj:
                tf.summary.scalar(key, metrics_obj[key][0])

            return metrics_obj


    def training(self, loss, learning_rate, decay_steps, decay_rate, staircase, var_list=tf.GraphKeys.TRAINABLE_VARIABLES):
        
        global_step = tf.Variable(self.global_step, name='global_step', trainable=False)

        # create learning_decay
        lr = tf.train.exponential_decay( learning_rate,
                                         global_step,
                                         decay_steps,
                                         decay_rate, staircase=staircase )

        tf.summary.scalar('2learning_rate', lr )

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step, var_list=var_list)

        return train_op

    def loss(self, logits, labels, class_weights=None):

        self.print_tensor_shape( logits, 'logits shape')
        self.print_tensor_shape( labels, 'labels shape')

        # return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
        return tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    
    def restore_variables(self):
        vars_train = tf.trainable_variables()
        # return [var for var in vars_train if 'generator' in var.name]
        return vars_train

    def prediction_type(self):
        return "image"

    def get_discriminator_vars(self):
        vars_train = tf.trainable_variables()

        vars_dis = [var for var in vars_train if 'discriminator' in var.name]

        for var in vars_dis:
          print('dis', var.name)

        return vars_dis

    def get_generator_vars(self):
        vars_train = tf.trainable_variables()

        vars_gen = [var for var in vars_train if 'generator' in var.name]        

        for var in vars_gen:
          print('gen', var.name)

        return vars_gen