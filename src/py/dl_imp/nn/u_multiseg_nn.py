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

        if("data_keys" in self.data_description 
            and len(self.data_description["data_keys"]) > 0
            and "max" in self.data_description[self.data_description["data_keys"][1]]):

            key_name_class = self.data_description["data_keys"][1]
            self.num_classes = int(self.data_description[self.data_description["data_keys"][1]]["max"])
        else:
            print("Setting the number of classes to 2", file=sys.stderr)
            self.num_classes = 2

    def inference(self, data_tuple=None, images=None, keep_prob=1, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):

    #   input: tensor of images
    #   output: tensor of computed logits
        if(data_tuple):
            images = data_tuple[0]

        self.print_tensor_shape(images, "images")

        shape = tf.shape(images)
        batch_size = shape[0]
        num_channels = images.get_shape().as_list()[-1]

        images = tf.layers.batch_normalization(images, training=is_training)

        conv1_0 = self.convolution2d(images, name="conv1_0_op", filter_shape=[5, 5, num_channels, 8], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        conv1_1 = self.convolution2d(conv1_0, name="conv1_1_op", filter_shape=[5, 5, 8, 8], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        pool1_0 = self.max_pool(conv1_1, name="pool1_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

        conv2_0 = self.convolution2d(pool1_0, name="conv2_0_op", filter_shape=[5, 5, 8, 16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        conv2_1 = self.convolution2d(conv2_0, name="conv2_1_op", filter_shape=[5, 5, 16, 16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        pool2_0 = self.max_pool(conv2_1, name="pool2_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)

        conv3_0 = self.convolution2d(pool2_0, name="conv3_0_op", filter_shape=[5, 5, 16, 32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        conv3_1 = self.convolution2d(conv3_0, name="conv3_1_op", filter_shape=[5, 5, 32, 32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        pool3_0 = self.max_pool(conv3_1, name="pool3_0_op", kernel=[1,5,5,1], strides=[1,2,2,1], ps_device=ps_device, w_device=w_device)
        
        pool3_0 = tf.nn.dropout( pool3_0, keep_prob)

        conv4_0 = self.convolution2d(pool3_0, name="conv4_0_op", filter_shape=[5, 5, 32, 64], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        conv4_1 = self.convolution2d(conv4_0, name="conv4_1_op", filter_shape=[5, 5, 64, 64], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        up_out_shape_4 = conv3_1.get_shape().as_list()
        up_out_shape_4[0] = batch_size
        up_conv4_0 = self.up_convolution2d(conv4_1, name="up_conv4_0_op", filter_shape=[5, 5, 32, 64], output_shape=up_out_shape_4, strides=[1,2,2,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        
        concat5_0 = tf.concat([up_conv4_0, conv3_1], -1)

        conv5_0 = self.convolution2d(concat5_0, name="conv5_0_op", filter_shape=[5, 5, 64, 32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        conv5_1 = self.convolution2d(conv5_0, name="conv5_1_op", filter_shape=[5, 5, 32, 32], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        up_out_shape_5 = conv2_1.get_shape().as_list()
        up_out_shape_5[0] = batch_size
        up_conv5_0 = self.up_convolution2d(conv5_1, name="up_conv5_0_op", filter_shape=[5, 5, 16, 32], output_shape=up_out_shape_5, strides=[1,2,2,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        
        concat6_0 = tf.concat([up_conv5_0, conv2_1], -1)

        conv6_0 = self.convolution2d(concat6_0, name="conv6_0_op", filter_shape=[5, 5, 32, 16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        conv6_1 = self.convolution2d(conv6_0, name="conv6_1_op", filter_shape=[5, 5, 16, 16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        up_out_shape_6 = conv1_1.get_shape().as_list()
        up_out_shape_6[0] = batch_size
        up_conv6_0 = self.up_convolution2d(conv6_1, name="up_conv6_0_op", filter_shape=[5, 5, 8, 16], output_shape=up_out_shape_6, strides=[1,2,2,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        
        concat7_0 = tf.concat([up_conv6_0, conv1_1], -1)

        conv7_0 = self.convolution2d(concat7_0, name="conv7_0_op", filter_shape=[5, 5, 16, 16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
        conv7_1 = self.convolution2d(conv7_0, name="conv7_1_op", filter_shape=[5, 5, 16, 16], strides=[1,1,1,1], padding="SAME", activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)

        final = self.convolution2d(conv7_1, name="final", filter_shape=[1, 1, 16, self.num_classes], strides=[1,1,1,1], padding="SAME", activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)

        return final

    def metrics(self, logits, data_tuple, name="metrics"):


        with tf.variable_scope(name):
            labels = data_tuple[1]
            labels = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
            labels = tf.reshape(labels, tf.shape(logits))

            metrics_obj = {}

            metrics_obj["ACCURACY"] = tf.metrics.accuracy(predictions=logits, labels=labels, name='accuracy')
            # metrics_obj["AUC"] = tf.metrics.auc(predictions=logits, labels=labels, name='auc')
            metrics_obj["FN"] = tf.metrics.false_negatives(predictions=logits, labels=labels, name='false_negatives')
            metrics_obj["FP"] = tf.metrics.false_positives(predictions=logits, labels=labels, name='false_positives')
            metrics_obj["TN"] = tf.metrics.true_negatives(predictions=logits, labels=labels, name='true_negatives')
            metrics_obj["TP"] = tf.metrics.true_positives(predictions=logits, labels=labels, name='true_positives')
            
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

    def loss(self, logits, data_tuple, images=None, class_weights=None):
        
        labels = data_tuple[1]
        labels = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
        labels = tf.reshape(labels, tf.shape(logits))

        self.print_tensor_shape( logits, 'logits shape')
        self.print_tensor_shape( labels, 'labels shape')

        logits_perm = tf.transpose(logits, perm=[3, 0, 1, 2]) 
        labels_perm = tf.transpose(labels, perm=[3, 0, 1, 2]) 

        self.print_tensor_shape( logits_perm, 'logits_perm shape')
        self.print_tensor_shape( labels_perm, 'labels_perm shape')

        return tf.reduce_sum(tf.map_fn(lambda num_class: self.iou(logits_perm[num_class], labels_perm[num_class]), tf.range(self.num_classes), dtype=tf.float32))

    def iou(self, logits, labels):

        self.print_tensor_shape( logits, 'logits iou shape')
        self.print_tensor_shape( labels, 'labels iou shape')

        shape = tf.shape(logits)
        batch_size = shape[0]

        labels_flat = tf.reshape(labels, [batch_size, -1])
        logits_flat = tf.reshape(logits, [batch_size, -1])

        intersection = 2.0 * tf.reduce_sum(logits_flat * labels_flat, axis=1) + 1e-7
        denominator = tf.reduce_sum(logits_flat, axis=1) + tf.reduce_sum(labels_flat, axis=1) + 1e-7

        return 1.0 - tf.reduce_mean(intersection / denominator)