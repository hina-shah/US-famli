
from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import importlib
import os
from datetime import datetime
import json
import glob

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='U network for segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--json', type=str, help='json file with the description of the inputs, generate it with tfRecords.py')
parser.add_argument('--nn', type=str, help='Type of neural network to use', default='u_nn')
parser.add_argument('--out', help='Output dirname for the model', default="./out")
parser.add_argument('--model', help='Output modelname, the output name will be <out directory>/model-<num step>', default="model")

parser.add_argument('--keep_prob', help='The probability that each element is kept during training', type=float, default=0.5)
parser.add_argument('--learning_rate', help='Learning rate, default=1e-5', type=float, default=1e-5)
parser.add_argument('--decay_rate', help='decay rate, default=0.96', type=float, default=0.96)
parser.add_argument('--decay_steps', help='decay steps, default=10000', type=int, default=1000)
parser.add_argument('--batch_size', help='Batch size for evaluation', type=int, default=8)
parser.add_argument('--num_epochs', help='Number of epochs', type=int, default=10)
parser.add_argument('--ps_device', help='Process device', type=str, default='/cpu:0')
parser.add_argument('--w_device', help='Worker device', type=str, default='/cpu:0')

args = parser.parse_args()

json_filename = args.json
outvariablesdirname = args.out
modelname = args.model
k_prob = args.keep_prob
learning_rate = args.learning_rate
decay_rate = args.decay_rate
decay_steps = args.decay_steps
batch_size = args.batch_size
num_epochs = args.num_epochs
ps_device = args.ps_device
w_device = args.w_device
neural_network = args.nn

nn = importlib.import_module(neural_network)

print('neural_network', neural_network)
print('json', json_filename)
print('keep_prob', k_prob)
print('learning_rate', learning_rate)
print('decay_rate', decay_rate)
print('decay_steps', decay_steps)
print('batch_size', batch_size)
print('num_epochs', num_epochs)
print('ps_device', ps_device)
print('w_device', w_device)

graph = tf.Graph()

with graph.as_default():

  iterator = nn.inputs(batch_size=batch_size,
    num_epochs=num_epochs,
    json_filename=json_filename)

  x, y_ = iterator.get_next()
  
  keep_prob = tf.placeholder(tf.float32)

  y_conv = nn.inference(x, keep_prob=keep_prob, is_training=True, ps_device=ps_device, w_device=w_device)

  # calculate the loss from the results of inference and the labels
  loss = nn.loss(y_conv, y_)

  tf.summary.scalar("loss", loss)

  train_step = nn.training(loss, learning_rate, decay_steps, decay_rate)

  metrics_eval = nn.metrics(y_conv, y_)

  summary_op = tf.summary.merge_all()

  with tf.Session() as sess:

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver = tf.train.Saver()
    # specify where to write the log files for import to TensorBoard
    now = datetime.now()
    summary_writer = tf.summary.FileWriter(os.path.join(outvariablesdirname, modelname + "-" + now.strftime("%Y%m%d-%H%M%S")), sess.graph)

    sess.run([iterator.initializer])
    step = 0

    while True:
      try:

        _, loss_value, summary, metrics = sess.run([train_step, loss, summary_op, metrics_eval], feed_dict={keep_prob: k_prob})

        if step % 100 == 0:
          print('OUTPUT: Step %d: loss = %.3f' % (step, loss_value))

          # output some data to the log files for tensorboard
          summary_writer.add_summary(summary, step)
          summary_writer.flush()

          metrics_str = '|'

          for metric in metrics:
            metrics_str += " %s = %.3f |" % (metric, metrics[metric][0])

          print(metrics_str)

          # less frequently output checkpoint files.  Used for evaluating the model
        if step % 1000 == 0:
          save_path = saver.save(sess, os.path.join(outvariablesdirname, modelname), global_step=step)

        step += 1

      except tf.errors.OutOfRangeError:
        break

    outmodelname = os.path.join(outvariablesdirname, modelname)
    print('Step:', step)
    print('Saving model:', outmodelname)
    saver.save(sess, outmodelname, global_step=step)
