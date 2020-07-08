
from pandas import read_csv, notna, notnull, DataFrame, to_numeric

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers

import os
from math import floor
import numpy as np

import logging
import utils
from argparse import ArgumentParser
from pathlib import Path

def df_to_dataset(dataframe, shuffle=True, batch_size=32, labels_name='ga_edd', show_data = False):
  dataframe = dataframe.copy()
  labels = dataframe.pop(labels_name)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe), seed=1234)
  if batch_size is not None:
    ds = ds.batch(batch_size)
  if show_data:
    cnt=0
    for x, y in ds:
      print(x, y)
      cnt+=1
      if cnt==1:
        break
  return ds

def split_df_to_ds(train_whole, test_size_in, batch_size, label):
  train, val = train_test_split(train_whole, test_size=test_size_in, random_state=1234)
  logging.info('Number of training samples {}'.format(len(train)))
  logging.info('Number of validation samples {}'.format(len(val)))
  train_ds = df_to_dataset(train, shuffle=False, batch_size=batch_size, labels_name=label)
  val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size, labels_name=label)
  return train_ds, val_ds

def create_fit_model(train_whole, feature_columns, layers_seq, batch_size, label,
                      val_size, rate, loss_fn, metrics_list, num_epochs):
  
  train_ds, val_ds = split_df_to_ds(train_whole, val_size, batch_size, label)
  try:
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    all_layers = [feature_layer]
    all_layers.extend(layers_seq)
    all_layers.append(layers.Dense(1))
    model = tf.keras.Sequential(all_layers)

    adam_opt = tf.keras.optimizers.Adam(learning_rate=rate)
    model.compile(adam_opt,
                  loss=loss_fn,
                  metrics=metrics_list
                  )

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=num_epochs,
              use_multiprocessing=True,
              verbose=0
    )

    val_loss = model.evaluate(val_ds)
    return model, val_loss
  except Exception as e:
    logging.error('Error training the trimester model {}'.format(e))
    return None, -1

##### Training for trimester classification model ###################
def train_trimester(train_whole, feature_columns, batch_size, label):
  logging.info('Trimester classification trainer')
  layers_list = [
      layers.Dense(4096, activation='relu'),  
      layers.Dense(2048, activation='relu'),  
      layers.Dense(1024, activation='relu'),  
      layers.Dense(512, activation='relu'),
      layers.Dense(256, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.1),
      layers.Dense(64, activation='relu')
  ]
  model_trim, val_loss = create_fit_model(train_whole, 
                                feature_columns, 
                                layers_list, 
                                batch_size, 
                                label, 
                                val_size= 0.1, 
                                rate= 0.0005, 
                                loss_fn= tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                                metrics_list= ['accuracy'], 
                                num_epochs= 20)

  if model_trim is None:
    logging.error('Error training the trimester model {}'.format(e))
    return None
  
  logging.info('Accuracy on the validation set is: {}'.format(val_loss))
  return model_trim


### 2nd trimester training/model routine ##############
def train_trimester_2(train_whole, feature_columns, batch_size, label):
  logging.info('2nd Trimester regresssion training')
  layers_list = [
      layers.Dense(256, activation='tanh'),
      layers.Dense(128, activation='tanh'),
      layers.Dense(64, activation='tanh'),
  ]
  model_2trim, val_loss = create_fit_model(train_whole, 
                                feature_columns, 
                                layers_list, 
                                batch_size, 
                                label, 
                                val_size= 0.1, 
                                rate= 0.001, 
                                loss_fn= tf.keras.losses.MeanSquaredError(), 
                                metrics_list=['MeanAbsoluteError', 'MeanSquaredError'], 
                                num_epochs= 100)

  if model_2trim is None:
    logging.error('Error training the trimester model {}'.format(e))
    return None
  
  logging.info("Validation set  metrics {}".format(val_loss))
  return model_2trim

### 3rd trimester training routine/model ############
def train_trimester_3(train_whole, feature_columns, batch_size, label):
  logging.info('3rd Trimester regresssion training')
  layers_list = [
      layers.Dense(1024, activation='relu'),  
      layers.Dense(512, activation='relu'),
      layers.Dense(256, activation='relu'),
  ]
  model_3trim, val_loss = create_fit_model(train_whole, 
                                feature_columns, 
                                layers_list, 
                                batch_size, 
                                label, 
                                val_size= 0.1, 
                                rate= 0.001, 
                                loss_fn= tf.keras.losses.MeanSquaredError(), 
                                metrics_list=['MeanAbsoluteError', 'MeanSquaredError'], 
                                num_epochs= 100)

  if model_3trim is None:
    logging.error('Error training the trimester model {}'.format(e))
    return None
  
  logging.info("Validation set  metrics {}".format(val_loss))
  return model_3trim

def train_general(train_whole, feature_columns, batch_size, label):
  
  logging.info('General regresssion training')
  layers_list = [
      layers.Dense(2048),
      layers.LeakyReLU(),  
      layers.Dense(1024),
      layers.LeakyReLU(),
      layers.Dense(512),
      layers.LeakyReLU(),
      layers.Dense(256),
      layers.LeakyReLU()
  ]
  model_g, val_loss = create_fit_model(train_whole, 
                                feature_columns, 
                                layers_list, 
                                batch_size, 
                                label, 
                                val_size= 0.1, 
                                rate= 0.001, 
                                loss_fn= tf.keras.losses.MeanSquaredError(), 
                                metrics_list=['MeanAbsoluteError', 'MeanSquaredError'], 
                                num_epochs= 100)

  if model_g is None:
    logging.error('Error training the trimester model {}'.format(e))
    return None
  
  logging.info("Validation set  metrics {}".format(val_loss))
  return model_g


def main(args):
    in_csv_path = Path(args.in_csv)
    y_name = args.y_var
    utils.setupLogFile(in_csv_path.parent)

    logging.info(' --- RUN for outvar {}, target {} ----- '.format(args.outvar, y_name))
    if not in_csv_path.exists():
        logging.error('Could not find the input file')
    
    try:
        dataframe = read_csv(str(in_csv_path))
        for column_name in dataframe.columns:
          if column_name.startswith('_'):
            dataframe.pop(column_name)
        
        for header in ['fl_1', 'bp_1', 'hc_1', 'ac_1', 'mom_age_edd', 'mom_weight_lb', 'mom_height_in']:
          r = max(dataframe[header]) - min(dataframe[header])
          dataframe[header] = (dataframe[header] - min(dataframe[header]))/r

        train_whole = dataframe[ (dataframe[y_name] != '.') & (notna(dataframe[y_name])) & (notnull(dataframe[y_name]))].copy()
        train_whole = train_whole.astype( {y_name : 'int32'})
       
        logging.info(' Number of trainig samples in the selected set: {}'.format(len(train_whole)))
        
        batch_size = 32 # A small batch sized is used for demonstration purposes
        feature_columns = []
        feature_names = []
        for header in ['fl_1', 'bp_1', 'hc_1', 'ac_1', 'mom_age_edd', 'mom_weight_lb', 'mom_height_in']:
            feature_columns.append(feature_column.numeric_column(header))
            feature_names.append(header)

        for header in ['hiv', 'current_smoker', 'former_smoker', 'chronic_htn', 'preg_induced_htn', 'diabetes', 'gest_diabetes']:
            col = feature_column.categorical_column_with_identity(header, 2)
            col = feature_column.indicator_column(col)
            feature_columns.append(col)
            feature_names.append(header)
        
        all_pred = [0]*len(dataframe)

        print('****** args.all_train is: {}'.format(args.all_train))
        if not args.all_train:
          trimester = train_whole['trimester'].values.tolist()        
          min_trim = min(trimester)
          max_trim = max(trimester)

          model_trim =  None 
          model_2trim = None
          model_3trim = None

          if min_trim == max_trim:
            # This data is only for one of the trimesters, run the training for one of them.
            logging.info('Training only for Trimester regression: {}'.format(min_trim+2))
            if min_trim == 0:
              model_2trim = train_trimester_2(train_whole, feature_columns, batch_size, y_name)
              if model_2trim is None:
                raise Exception('2nd trimester model empty')
            else:
              model_3trim = train_trimester_3(train_whole, feature_columns, batch_size, y_name)
              if model_3trim is None:
                raise Exception('3rd trimester model empty')
          else:
            model_trim = train_trimester(train_whole, feature_columns, batch_size, 'trimester')
            trim_2_df = train_whole[ train_whole['trimester'] == 0]
            model_2trim = train_trimester_2(trim_2_df, feature_columns, batch_size, y_name)
            trim_3_df = train_whole[ train_whole['trimester'] == 1]
            model_3trim = train_trimester_3(trim_3_df, feature_columns, batch_size, y_name)
            logging.info('-- done training for all three ')
            if model_trim is None or model_2trim is None and model_3trim is None:
              raise Exception('One of the models came back empty during the classification/regression phase')
            
          # Classify the dataset if this is a multi-trimester dataset
          if model_trim is not None and model_2trim is not None and model_3trim is not None:
            logging.info('Creating predictions for the full dataset')
            ds = df_to_dataset(dataframe, shuffle=False, batch_size=32, labels_name=y_name)
            ga_2trim = model_2trim.predict(ds)
            ga_3trim = model_3trim.predict(ds)
            
            ds = df_to_dataset(dataframe, shuffle=False, batch_size=32, labels_name='trimester')
            c_p = (model_trim.predict(ds) > 0).astype("int32")
            
            all_pred = [ g_2[0] if c==0 else g_3[0] for (g_2, g_3, c) in zip(ga_2trim, ga_3trim, c_p)  ]
            logging.info('Length of all predictions list is: {}'.format(len(all_pred)))
            
          elif min_trim == max_trim:
            ds = df_to_dataset(dataframe, shuffle=False, batch_size=32, labels_name=y_name)
            if min_trim == 0 and model_2trim is not None:
              all_pred = model_2trim.predict(ds)
            elif min_trim == 1 and model_3trim is not None:
              all_pred = model_3trim.predict(ds)
            else:
              logging.error('Either 2nd or 3rd trimester data is null')
          else:
            logging.error('We are in unknown territory, exiting')

        else: # Per trimester if/else
          model_g = train_general(train_whole, feature_columns, batch_size, y_name)
          ds = df_to_dataset(dataframe, shuffle=False, batch_size=32, labels_name=y_name)
          all_pred = model_g.predict(ds)

        logging.info('Creating output dataset')
        out_df = dataframe[['PatientID', 'filename', 'studydate']].copy()
        out_df[args.outvar] = all_pred
        out_path = in_csv_path.parent/(args.outvar + '.csv')
        logging.info('Should output to: {}'.format(out_path))  
        out_df.to_csv(out_path)
    except Exception as e:
        logging.error('Error: \n{}'.format(e))
        logging.error(e)

if __name__=="__main__":
# /Users/hinashah/famli/Groups/Restricted_access_data/Clinical_Data/EPIC/Dataset_B
    parser = ArgumentParser()
    parser.add_argument('--in_csv', type=str, help='Input file')
    parser.add_argument('--y_var', type=str, help='Name of the target variable')
    parser.add_argument('--outvar', type=str, help='Name of the output variable')
    parser.add_argument('--all_train', action='store_true', help='If included per trimester training will not be performed')
    args = parser.parse_args()

    main(args)