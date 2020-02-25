
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

output_param_group = parser.add_argument_group('Output')
output_param_group.add_argument('--out', help='Output dirname for the model', default="./out")
output_param_group.add_argument('--model', help='Output modelname, the output name will be <out directory>/model-<num step>', default="model")

train_param_group = parser.add_argument_group('Training parameters')
train_param_group.add_argument('--nn', type=str, help='Type of neural network to use', default='u_nn')
train_param_group.add_argument('--keep_prob', help='The probability that each element is kept during training', type=float, default=0.5)
train_param_group.add_argument('--learning_rate', help='Learning rate, default=1e-5', type=float, default=1e-5)
train_param_group.add_argument('--learning_rate_discriminator_mul', help='Factor for the learning rate for the discriminator (to make it slower/faster)', type=float, default=1)
train_param_group.add_argument('--decay_rate', help='decay rate, default=0.96', type=float, default=0.96)
train_param_group.add_argument('--decay_steps', help='decay steps, default=10000', type=int, default=1000)
train_param_group.add_argument('--staircase', help='staircase decay', type=bool, default=False)
train_param_group.add_argument('--batch_size', help='Batch size for evaluation', type=int, default=8)
train_param_group.add_argument('--num_epochs', help='Number of epochs', type=int, default=10)
train_param_group.add_argument('--buffer_size', help='Shuffle buffer size', type=int, default=1000)
train_param_group.add_argument('--ps_device', help='Process device', type=str, default='/cpu:0')
train_param_group.add_argument('--w_device', help='Worker device', type=str, default='/cpu:0')

continue_param_group = parser.add_argument_group('Continue training', 'Use a previously saved model to continue the training.')
continue_param_group.add_argument('--in_model', help='Input model name', default=None)
continue_param_group.add_argument('--in_step', type=int, help='Set the step number to start', default=0)

args = parser.parse_args()

json_filename = args.json
neural_network = args.nn
outvariablesdirname = args.out
modelname = args.model
k_prob = args.keep_prob
learning_rate = args.learning_rate
learning_rate_discriminator_mul = args.learning_rate_discriminator_mul
decay_rate = args.decay_rate
decay_steps = args.decay_steps
staircase = args.staircase
batch_size = args.batch_size
num_epochs = args.num_epochs
buffer_size = args.buffer_size
ps_device = args.ps_device
w_device = args.w_device

in_model = args.in_model
in_step = args.in_step

loss_value_g = 0

if(args.args):
  with open(args.args, "r") as jsf:
    json_args = json.load(jsf)

    json_filename = json_args["json"] if json_args["json"] else args.json
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
is_gan = "gan" in neural_network and "gan_encoder" not in neural_network
is_gan_encoder = "gan_encoder" in neural_network
is_ed = "ed_nn" in neural_network

if(in_step > 0):
  nn.set_global_step(in_step)

print('json', json_filename)
print('neural_network', neural_network)
if is_gan:
  print('using gan optimization scheme')
if is_gan_encoder:
  print('using gan_encoder optimization scheme')
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
    gen_x = nn.inference(data_tuple, keep_prob=keep_prob, is_training=True, ps_device=ps_device, w_device=w_device)

    images = data_tuple[0]

    real_y = nn.discriminator(images=images, keep_prob=keep_prob, num_labels=2, is_training=True, ps_device=ps_device, w_device=w_device)
    fake_y = nn.discriminator(images=gen_x, keep_prob=1, num_labels=2, is_training=True, reuse=True, ps_device=ps_device, w_device=w_device)
    
    # calculate the loss for the fake/generated images
    real_y_ = tf.one_hot(tf.ones([tf.shape(real_y)[0]], tf.int32), 2)
    fake_y_ = tf.one_hot(tf.zeros([tf.shape(fake_y)[0]], tf.int32), 2)

    # calculate the loss for the discriminator
    loss_d_real = nn.loss(real_y, real_y_)
    loss_d_fake = nn.loss(fake_y, fake_y_)
    loss_d = loss_d_real + loss_d_fake

    tf.summary.scalar("loss_d_real", loss_d_real)
    tf.summary.scalar("loss_d_fake", loss_d_fake)
    tf.summary.scalar("loss_d", loss_d)

    # calculate the loss for the generator, i.e., trick the discriminator
    loss_g = nn.loss(fake_y, real_y_)
    tf.summary.scalar("loss_g", loss_g)

    tf.summary.image('generated', gen_x)
    tf.summary.image('real', images)

    # setup the training operations
    with tf.variable_scope("train_discriminator") as scope:
      train_op_d = nn.training(loss_d, learning_rate * learning_rate_discriminator_mul, decay_steps, decay_rate, staircase, var_list=nn.get_discriminator_vars())
    with tf.variable_scope("train_generator") as scope:
      train_op_g = nn.training(loss_g, learning_rate, decay_steps, decay_rate, staircase, var_list=nn.get_generator_vars())

    metrics_eval = nn.metrics(fake_y, real_y)

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
      saver = tf.train.Saver()

      if(in_model):
        vars_restore = nn.restore_variables()
        for var in vars_restore:
          print('res', var.name)
        saver_in = tf.train.Saver(vars_restore)
        saver_in.restore(sess, in_model)
      # specify where to write the log files for import to TensorBoard
      now = datetime.now()

      summary_path = os.path.join(outvariablesdirname, modelname + "-" + now.strftime("%Y%m%d-%H%M%S"))
      summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
      outmodelname = summary_path

      sess.run([iterator.initializer])
      step = nn.get_global_step()

      while True:
        try:

          _d, _g, loss_value_d, loss_value_g, loss_value_d_fake, summary, metrics = sess.run([train_op_d, train_op_g, loss_d, loss_g, loss_d_fake, summary_op, metrics_eval], feed_dict={keep_prob: k_prob})
          
          if step % 100 == 0:
            print('OUTPUT: Step %d: loss_g = %.3f, loss_d = %.3f, loss_d_fake = %.3f' % (step, loss_value_g, loss_value_d, loss_value_d_fake))

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
            print('Saving model:', outmodelname + "-" + str(step))
            with open(os.path.join(outvariablesdirname, modelname + ".json"), "w") as f:
              args_dict = vars(args)
              args_dict["model"] = os.path.basename(outmodelname) + "-" + str(step)
              args_dict["description"] = nn.get_data_description()
              if 'args' in args_dict:
                del args_dict['args']
              f.write(json.dumps(args_dict))

          step += 1

        except tf.errors.OutOfRangeError:
          break

      print('Step:', step)
      print('Saving model:', os.path.join(outvariablesdirname, modelname + ".json"))
      saver.save(sess, outmodelname, global_step=step)

      with open(os.path.join(outvariablesdirname, modelname + ".json"), "w") as f:
        args_dict = vars(args)
        args_dict["model"] = os.path.basename(modelname) + "-" + str(step)
        args_dict["description"] = nn.get_data_description()
        if 'args' in args_dict:
          del args_dict['args']
        f.write(json.dumps(args_dict))
    
  else:
    # THIS IS THE STANDARD OPIMIZATION SCHEME FOR NETWORKS SUCH AS UNET, CLASSIFICATION OR LABEL MAPS

    if(is_gan_encoder):
      # with tf.variable_scope("encoder"):
      #   with tf.variable_scope("encoder_bn"):
      #     images = tf.layers.batch_normalization(data_tuple[0], training=True, trainable=False)
      images = data_tuple[0]
          
      encoded_x = nn.inference(images=images, keep_prob=keep_prob, is_training=True, ps_device=ps_device, w_device=w_device)
      y_conv = nn.generator(images=encoded_x, ps_device=ps_device, w_device=w_device)
      
      loss = nn.loss(y_conv, images, encoded_x)

      tf.summary.scalar("loss", loss)

      train_step = nn.training(loss, learning_rate, decay_steps, decay_rate, staircase)

      metrics_eval = nn.metrics(y_conv, images)

      tf.summary.image('generated', y_conv)
      tf.summary.image('real', images)

    else:
      y_conv = nn.inference(data_tuple, keep_prob=keep_prob, is_training=True, ps_device=ps_device, w_device=w_device)

      loss = nn.loss(y_conv, data_tuple)

      tf.summary.scalar("loss", loss)

      train_step = nn.training(loss, learning_rate, decay_steps, decay_rate, staircase)

      metrics_eval = nn.metrics(y_conv, data_tuple)

      if(is_ed):
        tf.summary.image('generated', y_conv)
        tf.summary.image('real', data_tuple[0])

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
      saver = tf.train.Saver(None)

      if(in_model):
        vars_restore = nn.restore_variables()
        for var in vars_restore:
          print('res', var.name)
        saver_in = tf.train.Saver(vars_restore)
        saver_in.restore(sess, in_model)
      # specify where to write the log files for import to TensorBoard
      now = datetime.now()

      summary_path = os.path.join(outvariablesdirname, modelname + "-" + now.strftime("%Y%m%d-%H%M%S"))
      summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
      outmodelname = summary_path

      sess.run([iterator.initializer])
      step = nn.get_global_step()

      while True:
        try:

          _, loss_value, summary, metrics = sess.run([train_step, loss, summary_op, metrics_eval], feed_dict={keep_prob: k_prob})

          if step % 100 == 0:
            print('OUTPUT: Step %d: loss = %.5f' % (step, loss_value))

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
            print('Saving model:', outmodelname + "-" + str(step))
            with open(os.path.join(outvariablesdirname, modelname + ".json"), "w") as f:
              args_dict = vars(args)
              args_dict["model"] = os.path.basename(modelname) + "-" + str(step)
              args_dict["description"] = nn.get_data_description()
              if 'args' in args_dict:
                del args_dict['args']
              f.write(json.dumps(args_dict))

          step += 1

        except tf.errors.OutOfRangeError:
          break
      
      print('Step:', step)
      print('Saving model:', os.path.join(outvariablesdirname, modelname + ".json"))
      saver.save(sess, outmodelname, global_step=step)

      with open(os.path.join(outvariablesdirname, modelname + ".json"), "w") as f:
        args_dict = vars(args)
        args_dict["model"] = os.path.basename(outmodelname) + "-" + str(step)
        args_dict["description"] = nn.get_data_description()
        if 'args' in args_dict:
          del args_dict['args']
        f.write(json.dumps(args_dict))
