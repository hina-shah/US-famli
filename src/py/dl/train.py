
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

in_group = parser.add_mutually_exclusive_group(required=True)
  
in_group.add_argument('--args', help='JSON file with arguments.', type=str)
in_group.add_argument('--json', type=str, help='json file with the description of the inputs, generate it with tfRecords.py')

parser.add_argument('--nn', type=str, help='Type of neural network to use', default='u_nn')
parser.add_argument('--out', help='Output dirname for the model', default="./out")
parser.add_argument('--model', help='Output modelname, the output name will be <out directory>/model-<num step>', default="model")

parser.add_argument('--keep_prob', help='The probability that each element is kept during training', type=float, default=0.5)
parser.add_argument('--learning_rate', help='Learning rate, default=1e-5', type=float, default=1e-5)
parser.add_argument('--decay_rate', help='decay rate, default=0.96', type=float, default=0.96)
parser.add_argument('--decay_steps', help='decay steps, default=10000', type=int, default=1000)
parser.add_argument('--staircase', help='staircase decay', type=bool, default=False)
parser.add_argument('--batch_size', help='Batch size for evaluation', type=int, default=8)
parser.add_argument('--num_epochs', help='Number of epochs', type=int, default=10)
parser.add_argument('--buffer_size', help='Shuffle buffer size', type=int, default=1000)
parser.add_argument('--ps_device', help='Process device', type=str, default='/cpu:0')
parser.add_argument('--w_device', help='Worker device', type=str, default='/cpu:0')

args = parser.parse_args()

json_filename = args.json
neural_network = args.nn
outvariablesdirname = args.out
modelname = args.model
k_prob = args.keep_prob
learning_rate = args.learning_rate
decay_rate = args.decay_rate
decay_steps = args.decay_steps
staircase = args.staircase
batch_size = args.batch_size
num_epochs = args.num_epochs
buffer_size = args.buffer_size
ps_device = args.ps_device
w_device = args.w_device

if(args.args):
  with open(args.args, "r") as jsf:
    json_args = json.load(jsf)

    json_filename = json_args["json_filename"] if json_args["json_filename"] else args.json
    neural_network = json_args["nn"] if json_args["nn"] else args.nn
    outvariablesdirname = json_args["out"] if json_args["out"] else args.out
    modelname = json_args["model"] if json_args["model"] else args.model
    k_prob = json_args["keep_prob"] if json_args["keep_prob"] else args.keep_prob
    learning_rate = json_args["learning_rate"] if json_args["learning_rate"] else args.learning_rate
    decay_rate = json_args["decay_rate"] if json_args["decay_rate"] else args.decay_rate
    decay_steps = json_args["decay_steps"] if json_args["decay_steps"] else args.decay_steps
    batch_size = json_args["batch_size"] if json_args["batch_size"] else args.batch_size
    num_epochs = json_args["num_epochs"] if json_args["num_epochs"] else args.num_epochs
    buffer_size = json_args["buffer_size"] if json_args["buffer_size"] else args.buffer_size
    ps_device = json_args["ps_device"] if json_args["ps_device"] else args.ps_device
    w_device = json_args["w_device"] if json_args["w_device"] else args.w_device


nn = importlib.import_module("nn." + neural_network).NN()
is_gan = "gan" in neural_network

print('json', json_filename)
print('neural_network', neural_network)
if is_gan:
  print('using gan optimization scheme')
print('out', outvariablesdirname)
print('keep_prob', k_prob)
print('learning_rate', learning_rate)
print('decay_rate', decay_rate)
print('decay_steps', decay_steps)
print('batch_size', batch_size)
print('num_epochs', num_epochs)
print('buffer_size', buffer_size)
print('ps_device', ps_device)
print('w_device', w_device)


graph = tf.Graph()

with graph.as_default():

  nn.set_data_description(json_filename=json_filename)
  iterator = nn.inputs(batch_size=batch_size,
    num_epochs=num_epochs, 
    buffer_size=buffer_size)

  data_tuple = iterator.get_next()
  
  keep_prob = tf.placeholder(tf.float32)

  if is_gan:
    # THIS IS THE GAN GENERATION NETWORK SCHEME
    # run the generator network on the 'fake/bad quality' input images (encode/decode)
    with tf.variable_scope("generator"):
      gen_x = nn.inference(x, keep_prob=keep_prob, is_training=True, ps_device=ps_device, w_device=w_device)

    with tf.variable_scope("discriminator") as scope:
      # run the discriminator network on the generated images
      gen_x = tf.layers.batch_normalization(gen_x, training=True)
      y_ = tf.layers.batch_normalization(y_, training=True)

      fake_y = nn.discriminator(gen_x, keep_prob=keep_prob, num_labels=2, is_training=True, ps_device=ps_device, w_device=w_device)

      scope.reuse_variables()
      real_y = nn.discriminator(y_, keep_prob=keep_prob, num_labels=2, is_training=True, ps_device=ps_device, w_device=w_device)
      

    # calculate the loss for the fake/generated images
    fake_y_ = tf.constant(np.zeros([batch_size], dtype=int))
    real_y_ = tf.constant(np.ones([batch_size], dtype=int))

    # calculate the loss for the discriminator
    loss_d = nn.loss(tf.concat([fake_y, real_y], axis=0), tf.concat([fake_y_, real_y_], axis=0))
    tf.summary.scalar("loss_d", loss_d)

    # calculate the loss for the generator, i.e., trick the discriminator
    loss_g = nn.loss(fake_y, real_y_)
    tf.summary.scalar("loss_g", loss_g)

    vars_train = tf.trainable_variables()

    vars_gen = [var for var in vars_train if 'generator' in var.name]        
    vars_dis = [var for var in vars_train if 'discriminator' in var.name]    

    for var in vars_gen:
      print('gen', var.name)

    for var in vars_dis:
      print('dis', var.name)

    # setup the training operations
    with tf.variable_scope("train_discriminator") as scope:
      train_op_d = nn.training(loss_d, learning_rate, decay_steps, decay_rate, vars_dis)
    # with tf.variable_scope("train_generator") as scope:
      train_op_g = nn.training(loss_g, learning_rate, decay_steps, decay_rate, vars_gen)

    metrics_eval = nn.metrics(gen_x, data_tuple)

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

          _d, _g, loss_value_d, loss_value_g, summary, metrics = sess.run([train_op_d, train_op_g, loss_d, loss_g, summary_op, metrics_eval], feed_dict={keep_prob: k_prob})

          if step % 100 == 0:
            print('OUTPUT: Step %d: loss_g = %.3f, loss_d = %.3f' % (step, loss_value_g, loss_value_d))

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

  else:
    # THIS IS THE STANDARD OPIMIZATION SCHEME FOR NETWORKS SUCH AS UNET, CLASSIFICATION OR LABEL MAPS
    y_conv = nn.inference(data_tuple, keep_prob=keep_prob, is_training=True, ps_device=ps_device, w_device=w_device)
    loss = nn.loss(y_conv, data_tuple)

    tf.summary.scalar("loss", loss)

    train_step = nn.training(loss, learning_rate, decay_steps, decay_rate, staircase)

    metrics_eval = nn.metrics(y_conv, data_tuple)

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
      saver = tf.train.Saver()
      # specify where to write the log files for import to TensorBoard
      now = datetime.now()

      summary_path = os.path.join(outvariablesdirname, modelname + "-" + now.strftime("%Y%m%d-%H%M%S"))
      summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

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

      outmodelname = summary_path
      print('Step:', step)
      print('Saving model:', outmodelname + "-" + str(step))
      saver.save(sess, outmodelname, global_step=step)

      with open(os.path.normpath(summary_path + "-" + str(step) + ".json"), "w") as f:
        args_dict = vars(args)
        args_dict["model"] = os.path.basename(outmodelname) + "-" + str(step)
        args_dict["description"] = nn.get_data_description()
        if 'args' in args_dict:
          del args_dict['args']
        f.write(json.dumps(args_dict))
