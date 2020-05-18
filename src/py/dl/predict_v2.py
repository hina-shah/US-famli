import numpy as np
import tensorflow as tf
import argparse
import importlib
import os
from datetime import datetime
import json
import glob
import itk
import sys
import csv


print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

in_group = parser.add_mutually_exclusive_group(required=True)
in_group.add_argument('--img', type=str, help='Input image for prediction')
in_group.add_argument('--dir', type=str, help='Directory with images for prediction')
in_group.add_argument('--csv', type=str, help='CSV file with images')
parser.add_argument('--csv_column', type=str, default='image', help='CSV column name (Only used if flag csv is used)')
parser.add_argument('--csv_root_path', type=str, default='', help='Replaces a root path directory to empty, this is use to recreate a directory structure in the output directory, otherwise, the output name will be the name in the csv (only if csv flag is used)')

parser.add_argument('--model', help='Directory of saved model format')
parser.add_argument('--class_prediction', help='If the model does class prediction', default=False)

parser.add_argument('--out', type=str, help='Output image, csv, or directory. If --dir flag is used the output image name will be the <Directory set in out flag>/<image filename in directory dir>', default="out")
parser.add_argument('--out_ext', type=str, help='Output extension for images', default='.nrrd')
parser.add_argument('--out_basename', type=bool, default=False, help='Keeps only the filename for the output, i.e, does not create a directory structure for the output image filename')
parser.add_argument('--ow', type=bool, help='Overwrite outputs', default=True)

args = parser.parse_args()

saved_model_path = args.model
out_name = args.out
out_ext = args.out_ext
class_prediction = args.class_prediction

filenames = []

if(args.img):
  print('image_name', args.img)
  fobj = {}
  fobj["img"] = args.img
  fobj["out"] = out_name
  if args.ow or not os.path.exists(fobj["out"]):
    filenames.append(fobj)
else:

  image_filenames = []
  replace_dir_name = ''
  if(args.dir):
    replace_dir_name = args.dir
    normpath = os.path.normpath("/".join([args.dir, '**', '*']))
    for img in glob.iglob(normpath, recursive=True):
      if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png"]]:
        image_filenames.append(img)
  elif(args.csv):
    replace_dir_name = args.csv_root_path
    with open(args.csv) as csvfile:
      csv_reader = csv.DictReader(csvfile)
      for row in csv_reader:
        image_filenames.append(row[args.csv_column])

  for img in image_filenames:
      fobj = {}
      fobj["img"] = img
      if(not class_prediction):
        image_dir_filename = img.replace(replace_dir_name, '')
        if(out_ext):
          image_dir_filename = os.path.splitext(image_dir_filename)[0] +  out_ext

        if(args.out_basename):
          image_dir_filename = os.path.basename(image_dir_filename)
          
        fobj["out"] = os.path.normpath("/".join([out_name, image_dir_filename]))

        if not os.path.exists(os.path.dirname(fobj["out"])):
          os.makedirs(os.path.dirname(fobj["out"]))

      if args.ow or not os.path.exists(fobj["out"]):
        filenames.append(fobj)


def image_read(filename):

  ImageType = itk.VectorImage[itk.F, 2]
  img_read = itk.ImageFileReader[ImageType].New(FileName=filename)
  img_read.Update()
  img = img_read.GetOutput()
  
  img_np = itk.GetArrayViewFromImage(img).astype(float)

  # Put the shape of the image in the json object if it does not exists. This is done for global information
  tf_img_shape = list(img_np.shape)
  if(tf_img_shape[0] == 1 and img.GetImageDimension() > 2):
    # If the first component is 1 we remove it. It means that is a 2D image but was saved as 3D
    tf_img_shape = tf_img_shape[1:]

  # This is the number of channels, if the number of components is 1, it is not included in the image shape
  # If it has more than one component, it is included in the shape, that's why we have to add the 1
  if(img.GetNumberOfComponentsPerPixel() == 1):
    tf_img_shape = tf_img_shape + [1]

  tf_img_shape = [1] + tf_img_shape

  return img, img_np, tf_img_shape


def image_save(img_obj, prediction):
  PixelDimension = prediction.shape[-1]
  Dimension = 2
  
  if(PixelDimension < 7):
    if(PixelDimension >= 3 and os.path.splitext(img_obj["out"])[1] not in ['.jpg', '.png']):
      ComponentType = itk.ctype('float')
      PixelType = itk.Vector[ComponentType, PixelDimension]
    elif(PixelDimension == 3):
      PixelType = itk.RGBPixel.UC
      prediction = np.absolute(prediction)
      prediction = np.around(prediction).astype(np.uint16)
    else:
      PixelType = itk.ctype('float')

    OutputImageType = itk.Image[PixelType, Dimension]
    out_img = OutputImageType.New()

  else:

    ComponentType = itk.ctype('float')
    OutputImageType = itk.VectorImage[ComponentType, Dimension]

    out_img = OutputImageType.New()
    out_img.SetNumberOfComponentsPerPixel(PixelDimension)
    
  size = itk.Size[Dimension]()
  size.Fill(1)
  prediction_shape = list(prediction.shape[0:-1])
  prediction_shape.reverse()

  for i, s in enumerate(prediction_shape):
    size[i] = s

  index = itk.Index[Dimension]()
  index.Fill(0)

  RegionType = itk.ImageRegion[Dimension]
  region = RegionType()
  region.SetIndex(index)
  region.SetSize(size)
  
  # out_img.SetRegions(img.GetLargestPossibleRegion())
  out_img.SetRegions(region)
  out_img.SetDirection(img.GetDirection())
  out_img.SetOrigin(img.GetOrigin())
  out_img.SetSpacing(img.GetSpacing())
  out_img.Allocate()

  out_img_np = itk.GetArrayViewFromImage(out_img)
  out_img_np.setfield(np.reshape(prediction, out_img_np.shape), out_img_np.dtype)

  print("Writing:", img_obj["out"])
  writer = itk.ImageFileWriter.New(FileName=img_obj["out"], Input=out_img)
  writer.UseCompressionOn()
  writer.Update()


if(int(tf.__version__.split('.')[0]) > 1):
  model = tf.keras.models.load_model(saved_model_path)
  model.summary()
  for layer in model.layers:
    if("batch_normalization" in layer.name):
      layer.training = True

  for img_obj in filenames:
      img, img_np, tf_img_shape = image_read(img_obj["img"])
      print(np.linalg.norm(img_np))
      prediction = tf.sigmoid(model.predict(np.reshape(img_np, (1,4096))))

      prediction = np.array(prediction[0])
      image_save(img_obj, prediction)

else:
  with tf.Session() as sess:

    loaded = tf.saved_model.load(sess=sess, tags=[tf.saved_model.SERVING], export_dir=saved_model_path)

    for img_obj in filenames:
      img, img_np, tf_img_shape = image_read(img_obj["img"])

      prediction = sess.run(
          'output_y:0',
          feed_dict={
              'input_x:0': np.reshape(img_np, (1,) + img_np.shape)
          }
      )

      prediction = np.array(prediction[0])
      image_save(img_obj, prediction)
    
