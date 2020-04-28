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
                self.out_channels = self.num_channels
                self.value_range = [self.data_description[data_keys[0]]["min"], self.data_description[data_keys[0]]["max"]]

    def up_conv_block(self, x0, in_filters, out_filters, cross_block, block='a', is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

        x = self.convolution2d(x0, name= block + "_conv1_op", filter_shape=[1,1,in_filters,out_filters[0]], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, initializer=tf.random_normal_initializer(mean=0,stddev=0.01), ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)

        out_shape=tf.shape(cross_block)
        x = self.up_convolution2d(x, name=block + "_up_conv1_op", filter_shape=[3,3,out_filters[1],out_filters[0]], output_shape=out_shape, strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, initializer=tf.random_normal_initializer(mean=0,stddev=0.01), ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)

        x = tf.concat([cross_block, x], -1)
        x = self.convolution2d(x, name=block + "_conv2_op", filter_shape=[1,1,x.get_shape().as_list()[-1],out_filters[2]], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, initializer=tf.random_normal_initializer(mean=0,stddev=0.01), ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)

        shortcut = self.up_convolution2d(x0, name=block + "_up_conv2_op", filter_shape=[3,3,out_filters[2],in_filters], output_shape=out_shape, strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, initializer=tf.random_normal_initializer(mean=0,stddev=0.01), ps_device=ps_device, w_device=w_device)
        shortcut = tf.layers.batch_normalization(shortcut, training=is_training)

        x = tf.math.add(x, shortcut)
        x = tf.nn.leaky_relu(x)

        return x

    def upblock(self, x0, out_filters, stride=2, ps_device="/cpu:0", w_device="/gpu:0"):

        shape = tf.shape(x0)
        batch_size = shape[0]

        output_shape = x0.get_shape().as_list()
        in_filters = output_shape[-1]
        output_shape = [batch_size,output_shape[1]*stride,output_shape[2]*stride,out_filters]

        x = self.up_convolution2d(x0, name="up_conv1", filter_shape=[1 + stride, 1 + stride,out_filters,in_filters], output_shape=output_shape, strides=[1,stride,stride,1], padding="SAME", use_bias=False, activation=None, initializer=tf.random_normal_initializer(mean=0,stddev=0.01), ps_device=ps_device, w_device=w_device)
        x = tf.layers.batch_normalization(x, training=True)
        x = tf.nn.leaky_relu(x)
        
        return x

    def higher_v2(self, x, reuse=False, is_training=False, keep_prob=1, ps_device="/cpu:0", w_device="/gpu:0"):
        with tf.variable_scope("ued_resnet") as scope:

            if(reuse):
                scope.reuse_variables()

            with tf.variable_scope("start"):
                x = tf.layers.batch_normalization(x, training=is_training)
                x = self.upblock(x, self.num_channels, stride=8, ps_device=ps_device, w_device=w_device)
                x0 = x

            with tf.variable_scope("block0"):
                x = self.convolution2d(x0, name="conv0_0_op", filter_shape=[3,3,self.num_channels,64], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)
                x = tf.nn.dropout(x, keep_prob)
                block0 = x
                
                # block0_0_shape = tf.shape(x)
                # x = self.convolution2d(x, name="conv1_0_op", filter_shape=[3,3,32,64], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)

            with tf.variable_scope("block1"):
                x = self.conv_block(x, 64, [64,64,128], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 64, [32,32,64], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 128, [64,64,128], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # block1 = tf.nn.dropout(x, keep_prob)
                # x = self.convolution2d(x, name="conv1_0_op", filter_shape=[3,3,64,128], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)
                # block0_0_shape = tf.shape(x)
                # x = self.convolution2d(x, name="conv1_0_op", filter_shape=[3,3,32,64], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)
                x = tf.nn.dropout(x, keep_prob)
                block1 = x
                
                

            with tf.variable_scope("block2"):
                x = self.conv_block(x, 128, [128,128,256], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 128, [64,64,128], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 128, [64,64,128], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 256, [128,128,256], block='d', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = tf.nn.dropout(x, keep_prob)
                block2 = x
                

            with tf.variable_scope("block3"):
                x = self.conv_block(x, 256, [256,256,512], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 256, [128,128,256], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 256, [128,256,256], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 256, [128,128,256], block='d', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 512, [256,256,512], block='e', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # block3 = tf.nn.dropout(x, keep_prob)
                x = tf.nn.dropout(x, keep_prob)

            # with tf.variable_scope("block4"):
            #     x = self.conv_block(x, 512, [512,512,1024], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
            #     x = self.identity_block(x, 1024, [512,512,1024], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
            #     # x = self.identity_block(x, 1024, [512,512,1024], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
            #     # x = self.identity_block(x, 1024, [512,512,1024], block='d', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
            #     # x = self.identity_block(x, 1024, [512,512,1024], block='e', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
            #     # x = self.identity_block(x, 1024, [512,512,1024], block='f', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
            #     x = tf.nn.dropout(x, keep_prob)

            # with tf.variable_scope("up_block4"):
            #     x = self.identity_block(x, 1024, [512,512,1024], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
            #     # x = self.identity_block(x, 1024, [512,512,1024], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
            #     # x = self.identity_block(x, 1024, [512,512,1024], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
            #     # x = self.identity_block(x, 1024, [512,512,1024], block='d', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
            #     # x = self.identity_block(x, 1024, [512,512,1024], block='e', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
            #     x = self.up_conv_block(x, 1024, [512,512,512], block='f', cross_block=block3, is_training=is_training, ps_device=ps_device, w_device=w_device)
            #     x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("up_block3"):
                # x = self.identity_block(x, 256, [128,128,256], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 256, [128,128,256], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 256, [128,128,256], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 512, [256,256,512], block='d', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.up_conv_block(x, 512, [256,256,256], block='e', cross_block=block2, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("up_block2"):
                # x = self.identity_block(x, 128, [64,64,128], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 128, [64,64,128], block='b', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 256, [128,128,256], block='c', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.up_conv_block(x, 256, [128,128,128], block='d', cross_block=block1, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = tf.nn.dropout(x, keep_prob)


            with tf.variable_scope("up_block1"):
                # x = self.identity_block(x, 64, [32,32,64], block='a', activation=tf.nn.leaky_relu, is_training=is_training, ps_device=ps_device, w_device=w_device)
                # x = self.identity_block(x, 128, [64,64,128], block='b', activation=tf.nn.leaky_relu,is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = self.up_conv_block(x, 128, [64,64,64], block='c', cross_block=block0, is_training=is_training, ps_device=ps_device, w_device=w_device)
                x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("up_block_final"):
                x = self.up_convolution2d(x, name="up_conv1_op", filter_shape=[3,3,self.out_channels,64], output_shape=tf.shape(x0), strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)
                # x = self.up_convolution2d(x, name="up_conv2_op", filter_shape=[3,3,1,32], output_shape=tf.shape(x0), strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.nn.tanh(x)
                x = tf.math.multiply(tf.math.add(x, 1), 127.5)
            return x

    def higher(self, x, reuse=False, is_training=False, keep_prob=1, ps_device="/cpu:0", w_device="/gpu:0"):
        with tf.variable_scope("higher") as scope:

            if(reuse):
                scope.reuse_variables()
            
            with tf.variable_scope("block0"):
                x = self.upblock(x, 256, ps_device=ps_device, w_device=w_device)
                x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("block1"):
                x = self.upblock(x, 128, ps_device=ps_device, w_device=w_device)
                x = tf.nn.dropout(x, keep_prob)

            with tf.variable_scope("block2"):
                shape = tf.shape(x)
                batch_size = shape[0]
                output_shape = x.get_shape().as_list()
                output_shape = [batch_size,output_shape[1]*2,output_shape[2]*2,self.out_channels]
                x = self.up_convolution2d(x, name="up_conv1_op", filter_shape=[3,3,self.out_channels,128], output_shape=output_shape, strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.nn.tanh(x)
                x = tf.math.multiply(tf.math.add(x, 1), 127.5)
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
            
            self.print_tensor_shape(x, "input_x")

            with tf.variable_scope("block0"):
                x = self.matmul(x, 4*4*1024, name='project', activation=None, initializer=tf.random_normal_initializer(mean=0,stddev=0.01), ps_device=ps_device, w_device=w_device)
                x = tf.reshape(x, [batch_size, 4, 4, 1024])
                x = tf.layers.batch_normalization(x, training=True)
                x = tf.nn.leaky_relu(x)

            with tf.variable_scope("block1"):
                x = self.upblock(x, 512, ps_device=ps_device, w_device=w_device)

            with tf.variable_scope("block2"):
                x = self.upblock(x, 256, ps_device=ps_device, w_device=w_device)

            with tf.variable_scope("block3"):
                x = self.upblock(x, 128, ps_device=ps_device, w_device=w_device)
            
            with tf.variable_scope("block4"):
                x = self.up_convolution2d(x, name="up_conv1", filter_shape=[3,3,self.out_channels,128], output_shape=[batch_size,64,64,self.out_channels], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, initializer=tf.random_normal_initializer(mean=0,stddev=0.01), ps_device=ps_device, w_device=w_device)
                x = tf.nn.tanh(x)
                x = tf.math.multiply(tf.math.add(x, 1), 127.5)

        return self.higher(x, is_training=True, keep_prob=keep_prob, ps_device=ps_device, w_device=w_device)

    def discriminator(self, images=None, data_tuple=None, num_labels=2, keep_prob=1, is_training=False, reuse=False, ps_device="/cpu:0", w_device="/gpu:0"):
    #   input: tensor of images
    #   output: tensor of computed logits
        if(data_tuple):
            images = data_tuple[1]
        self.print_tensor_shape(images, "images")

        shape = tf.shape(images)
        batch_size = shape[0]
        
        with tf.variable_scope("discriminator_512") as scope:
            if(reuse):
                scope.reuse_variables()

            x = tf.layers.batch_normalization(images, training=is_training)

            with tf.variable_scope("block0"):
                x = self.convolution2d(x, name="conv1", filter_shape=[3,3,self.num_channels,16], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)
                x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
                # x = self.convolution2d(x, name="conv2", filter_shape=[5,5,16,16], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)
                
            with tf.variable_scope("block1"):
                x = self.convolution2d(x, name="conv1", filter_shape=[3,3,16,32], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)
                x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
                # x = self.convolution2d(x, name="conv2", filter_shape=[5,5,32,32], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)

            with tf.variable_scope("block2"):
                x = self.convolution2d(x, name="conv1", filter_shape=[3,3,32,64], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)
                x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
                # x = self.convolution2d(x, name="conv2", filter_shape=[5,5,64,64], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)

            with tf.variable_scope("block3"):
                x = self.convolution2d(x, name="conv1", filter_shape=[3,3,64,128], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)            
                x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
                # x = self.convolution2d(x, name="conv2", filter_shape=[3,3,128,128], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)

            with tf.variable_scope("block4"):
                x = self.convolution2d(x, name="conv1", filter_shape=[3,3,128,256], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)            
                x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
                # x = self.convolution2d(x, name="conv2", filter_shape=[3,3,256,256], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                # x = tf.layers.batch_normalization(x, training=is_training)
                # x = tf.nn.leaky_relu(x)

            with tf.variable_scope("block5"):
                x = self.convolution2d(x, name="conv1", filter_shape=[3,3,256,512], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)            
                x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
            #     # x = self.convolution2d(x, name="conv2", filter_shape=[3,3,512,512], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
            #     # x = tf.layers.batch_normalization(x, training=is_training)
            #     # x = tf.nn.leaky_relu(x)

            with tf.variable_scope("block6"):
                x = self.convolution2d(x, name="conv1", filter_shape=[3,3,512,1024], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)            
                x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
            #     # x = self.convolution2d(x, name="conv2", filter_shape=[3,3,1024,1024], strides=[1,2,2,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
            #     # x = tf.layers.batch_normalization(x, training=is_training)
            #     # x = tf.nn.leaky_relu(x)

            with tf.variable_scope("fc"):
                # kernel_size = x.get_shape().as_list()
                # kernel_size[0] = 1
                # kernel_size[-1] = 1
                # x = self.avg_pool(x, name="avg_pool_op", kernel=kernel_size, strides=kernel_size, ps_device=ps_device, w_device=w_device)
                x = tf.reshape(x, (batch_size, 4*4*1024))
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

    def discriminator_lr(self, images=None, data_tuple=None, num_labels=2, keep_prob=1, is_training=False, reuse=False, ps_device="/cpu:0", w_device="/gpu:0"):
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
                x = self.convolution2d(x, name="conv1", filter_shape=[3,3,self.num_channels,128], strides=[1,1,1,1], padding="SAME", use_bias=False, activation=None, ps_device=ps_device, w_device=w_device)
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.leaky_relu(x)
                x = self.avg_pool(x, name="avg_pool_op", kernel=[1,3,3,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

            return x

    def loss_high(self, logits, labels):

        shape = tf.shape(logits)
        batch_size = shape[0]

        labels_conv = self.discriminator_lr(images=labels)
        # labels_conv = tf.reshape(labels_conv, [batch_size, -1])
        logits_conv = self.discriminator_lr(images=logits, reuse=True)
        # logits_conv = tf.reshape(logits_conv, [batch_size, -1])
        
        # labels = tf.math.subtract(tf.math.multiply(tf.math.divide(tf.math.subtract(labels, self.data_description["image"]["min"]), self.data_description["image"]["max"] - self.data_description["image"]["min"]), 2.0), 1.0)
        # labels = tf.layers.batch_normalization(labels, training=True)

        # logits_flat = tf.reshape(logits, [batch_size, -1])
        # labels_flat = tf.reshape(labels, [batch_size, -1])

        # return self.emd(labels_conv, logits_conv)
        return self.emd(logits_conv, labels_conv)

    
    def restore_variables(self, restore_all=True):
        vars_train = tf.trainable_variables()
        if(restore_all):
            return vars_train
        return [var for var in vars_train if 'generator' in var.name and 'generator_512' not in var.name or 'discriminator' in var.name and 'discriminator_512' not in var.name]

    def prediction_type(self):
        return "image"

    def get_discriminator_vars(self):
        vars_train = tf.trainable_variables()

        vars_dis = [var for var in vars_train if 'discriminator_512' in var.name]

        for var in vars_dis:
          print('dis', var.name)

        return vars_dis

    def get_generator_vars(self):
        vars_train = tf.trainable_variables()

        vars_gen = [var for var in vars_train if 'higher' in var.name]

        for var in vars_gen:
          print('gen', var.name)

        return vars_gen