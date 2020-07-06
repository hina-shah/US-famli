# -*- coding: utf-8 -*-

from pandas import read_csv
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers

import os
import datetime
from math import floor
import csv

from argparse import ArgumentParser

print(tf.__version__)
print(pd.__version__)


def df_to_dataset(dataframe, shuffle=True, batch_size=32, labels_name='ga_edd', show_data = False):
  dataframe = dataframe.copy()
  labels = dataframe.pop(labels_name)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe), seed=1234)
  ds = ds.batch(batch_size)
  if show_data:
    print('----- TF Dataset -------')
    cnt=0
    for x, y in ds:
      print(x, y)
      cnt+=1
      if cnt==1:
        break
  return ds

def run_training(train_ds, val_ds, test_ds, layer_nodes_count, feature_columns, 
              activation = 'relu', dropout = 0.0, verb = 1):
  mse = -1.0
  val_mse = -1.0
  train_mse = -1.0

  #logdir = os.path.join("logs", "overall_train")
  #tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

  feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

  ip_layers = [feature_layer]
  for layer_count in layer_nodes_count:

    if activation != 'leakyrelu':
      ip_layers.append(layers.Dense(layer_count, activation))
    else:
      ip_layers.append(layers.Dense(layer_count))
      ip_layers.append(layers.LeakyReLU())
    if dropout > 0.1:
      ip_layers.append(layers.Dropout(dropout))
  ip_layers.append(layers.Dense(1))

  model_tr = tf.keras.Sequential(ip_layers)

  adam_opt = tf.keras.optimizers.Adam(learning_rate=0.001)

  try:
    model_tr.compile(adam_opt,
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=['MeanAbsoluteError', 'MeanSquaredError'])

    model_tr.fit(train_ds,
            validation_data=val_ds,
            epochs=200,
           #callbacks=[tensorboard_callback], 
            #use_multiprocessing=True, 
            verbose = verb
            )
    train_mse = model_tr.evaluate(train_ds)[0]
    val_mse = model_tr.evaluate(val_ds)[0]
    mse = model_tr.evaluate(test_ds)[0]
  except Exception as e:
    print('Had an error somewhere, resetting return mse {}'.format(e))
    
  return train_mse, val_mse, mse

def main(args):

  try:
    dataframe = read_csv(args.in_csv)
  except IOError as e:
    print('Error reading the input csv file')
    return
  
  for header in ['fl_1', 'bp_1', 'hc_1', 'ac_1', 'mom_age_edd', 'mom_weight_lb', 'mom_height_in']:
    r = max(dataframe[header]) - min(dataframe[header])
    dataframe[header] = (dataframe[header] - min(dataframe[header]))/r

  train_whole, test = train_test_split(dataframe, test_size=0.2, random_state=1234)
  print(len(test), 'testing samples')
  train, val = train_test_split(train_whole, test_size=0.1, random_state=1234)
  print(len(train), 'train examples')
  print(len(val), 'validation examples')

  feature_columns = []

  for header in ['fl_1', 'bp_1', 'hc_1', 'ac_1', 'mom_age_edd', 'mom_weight_lb', 'mom_height_in']:
    feature_columns.append(feature_column.numeric_column(header))

  for header in ['hiv', 'current_smoker', 'former_smoker', 'chronic_htn', 'preg_induced_htn', 'diabetes', 'gest_diabetes']:
      col = feature_column.categorical_column_with_identity(header, 2)
      col = feature_column.indicator_column(col)
      feature_columns.append(col)

  # This is full dataset training:
  batch_size = 32
  train_ds = df_to_dataset(train, shuffle=False, batch_size=batch_size)
  val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
  test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

  activations = ['relu', 'leakyrelu', 'tanh']
  layer_counts = [
                  [64],
                  [128],
                  [256],
                  [512],
                  [1024],
                  [2048],
                  [4096],
                  [64, 64, 64],
                  [128, 128, 128],
                  [256, 256, 256],
                  [512, 512, 512],
                  [1024, 1024, 1024],
                  [128, 64],
                  [256, 128, 64],
                  [512, 256, 128],
                  [1024, 512, 256],
                  [2048, 1024, 512, 256],
                  [4096, 2048, 1024, 512, 256],
  ]

  dropouts = [0.0, 0.1]

  process_config = [ (t, s, b) for t in activations
                                  for s in layer_counts 
                                  for b in dropouts ]
  csv_rows = []
  len(process_config)
  for i, config in enumerate(process_config):
    print('******* Processing config {}/{}'.format(i, len(process_config)))
    train_mse, val_mse, mse = run_training(train_ds, val_ds, test_ds, config[1], 
                                          feature_columns, 
                                          activation = config[0], 
                                          dropout = config[2], verb = 0)
    print('******* For config: {} \n Train MSE: {} Val MSE: {} Test MSE: {}'.format(config, train_mse, val_mse, mse))
    out_row = {}
    out_row['activation'] = config[0]
    out_row['config'] = ' '.join(str(e) for e in config[1])
    out_row['dropout'] = config[2]
    out_row['train_mse'] = train_mse
    out_row['val_mse'] = val_mse
    out_row['test_mse'] = mse

    try:
      fields = out_row.keys()
      if i==0:
        f = open(args.out_csv, 'w')
        csvwriter = csv.DictWriter(f, fields)
        csvwriter.writeheader()
      else:
        f = open(args.out_csv, 'a')
        csvwriter = csv.DictWriter(f, fields)
      csvwriter.writerow(out_row)
      f.close()
    except Exception as e:
      print('Error writing out the file: {}'.format(e))
  print('----- DONE ------------')


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--in_csv', type=str, help='Input data', required=True)
    parser.add_argument('--out_csv', type=str, help='Output data', required=True)
    args = parser.parse_args()

    main(args)