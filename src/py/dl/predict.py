
from __future__ import print_function
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

in_group_model = parser.add_mutually_exclusive_group(required=True)
in_group_model.add_argument('--json', help='JSON file with model description, created by train.py')

in_group_model.add_argument('--model', help='Model created by train.py')
parser.add_argument('--nn', type=str, help='neural network type, use when flag --model is used', default=None)
parser.add_argument('--data_description', type=str, help='JSON file created by tfRecords.py use when --model and --nn flag are used', default=None)

parser.add_argument('--out', type=str, help='Output image, csv, or directory. If --dir flag is used the output image name will be the <Directory set in out flag>/<image filename in directory dir>', default="out")
parser.add_argument('--out_ext', type=str, help='Output extension for images', default='.nrrd')
parser.add_argument('--out_basename', type=bool, default=False, help='Keeps only the filename for the output, i.e, does not create a directory structure for the output image filename')
parser.add_argument('--ow', type=int, help='Overwrite outputs', default=1)
parser.add_argument('--resize', nargs="+", type=int, help='Resize images during prediction, useful when doing whole directories with images of diferent sizes. This is needed to set the value of the placeholder for tensorflow. e.x. 1500 1500. The image will be resized for the prediction but the original image size will be stored. Do not include the channels/pixel components in the resize parameters', default=None)
parser.add_argument('--resize_prediction', type=bool, help='If the resize flag is used and resize_prediction is set to True, the output/prediction will be resized as well. Otherwise, the output of the prediction will be used as the default shape', default=False)
parser.add_argument('--ps_device', help='Process device', type=str, default='/cpu:0')
parser.add_argument('--w_device', help='Worker device', type=str, default='/cpu:0')

args = parser.parse_args()

json_model_name = args.json
out_name = args.out
out_ext = args.out_ext
resize_shape = args.resize
resize_prediction = args.resize_prediction
ps_device = args.ps_device
w_device = args.w_device
data_description = {} 

if json_model_name is not None:
  with open(json_model_name, "r") as f:
    model_description = json.load(f)
    if "description" in model_description:
      data_description = model_description["description"]
    model_name = os.path.join(os.path.dirname(json_model_name), model_description["model"])
    neural_network = model_description["nn"]
elif args.model is not None and args.nn is not None and args.data_description:
  model_name = args.model
  neural_network = args.nn
  with open(args.data_description, "r") as f:
    data_description = json.load(f)
else:
  print("Set the --json or the --model and --nn and --data_description parameters", file=sys.stderr)
  sys.exit(1)

print('json', json_model_name)
print('model_name', model_name)
print('neural_network', neural_network)
is_gan = "gan" in neural_network
if is_gan:
  print('Using gan scheme')
print('ps_device', ps_device)
print('w_device', w_device)
print('ow', args.ow)
print('out', out_name)
print('out_ext', out_ext)

nn = importlib.import_module("nn." + neural_network).NN()

if(data_description):
  nn.set_data_description(data_description=data_description)

class_prediction = False
class_prediction_arr = []
if("enumerate" in data_description):
  class_prediction = True
  class_obj = {}
  enumerate_obj = data_description[data_description["enumerate"]]["class"]
  for key in enumerate_obj:
    class_obj[enumerate_obj[key]] = key 

  print(class_obj)
prediction_arr = []

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


if "slice" in data_description and data_description["slice"]:

  filenames_cp = filenames[:]
  filenames = []
  for slimg in filenames_cp:

    slimg_base = os.path.splitext(os.path.basename(slimg["img"]))[0]

    slice_obj = {}
    slice_obj["img"] = [slimg]
    slice_obj["out_csv_headers"] = ["img"]
    slice_obj["out"] = os.path.join(args.out, slimg_base)
    slice_obj["out_csv"] = os.path.join(args.out, slimg_base, "slice.csv")
    slice_obj["json_desc"] = os.path.join(args.out, slimg_base, "slice.json")

    # We convert the dictionary to a namedtuple, a.k.a, python object, i.e., argparse object
    slice_args = namedtuple("Slice", slice_obj.keys())(*slice_obj.values())
    # Call the main of the script
    nrrd3D_2D.main(slice_args)

    # We read the saved data and append the properties of the 3D image to all slices
    with open(slice_obj["out_csv"]) as csvfileslice:
      csv_slice_reader = csv.DictReader(csvfileslice)
      for slice_row in csv_slice_reader:
        # We now have a csv_rows list with slices instead of 3D volumes
        fobj = {}
        fobj["img"] = slice_row["image"]
        img_out_name = os.path.basename(slice_row["image"])
        if(out_ext):
          img_out_name = os.path.splitext()[0] + out_ext
        fobj["out"] = os.path.join(args.out, slimg_base, "predict", img_out_name)
        filenames.append(fobj)

if len(filenames) == 0:
  print("No images found with extensions", [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".jpg", ".png"], file=sys.stderr)
  sys.exit()

def image_read(filename):
  img_read = itk.ImageFileReader.New(FileName=filename)
  img_read.Update()
  img = img_read.GetOutput()
  
  img_np = itk.GetArrayViewFromImage(img).astype(float)

  # Put the shape of the image in the json object if it does not exists. This is done for global information
  tf_img_shape = list(img_np.shape)
  if(tf_img_shape[0] == 1):
    # If the first component is 1 we remove it. It means that is a 2D image but was saved as 3D
    tf_img_shape = tf_img_shape[1:]

  # This is the number of channels, if the number of components is 1, it is not included in the image shape
  # If it has more than one component, it is included in the shape, that's why we have to add the 1
  if(img.GetNumberOfComponentsPerPixel() == 1):
    tf_img_shape = tf_img_shape + [1]

  tf_img_shape = [1] + tf_img_shape

  return img, img_np, tf_img_shape

try:
  _itkimg, _imgnp, tf_img_shape = image_read(filenames[0]["img"])
except Exception as e:
  print("Error reading image to get shape info for prediction", filenames[0]["img"], e, file=sys.stderr)
  sys.exit()

if("resize" in data_description):
  resize_shape = data_description["resize"]

if(resize_shape):
  tf_img_shape = [1] + resize_shape + [tf_img_shape[-1]]

print("resize_shape", resize_shape)
print("resize_prediction", resize_prediction)
print("tf_img_shape", tf_img_shape)

graph = tf.Graph()

with graph.as_default():

  x = tf.placeholder(tf.float32,shape=tf_img_shape)
  
  keep_prob = tf.placeholder(tf.float32)

  if is_gan:
    with tf.variable_scope("generator"):
      y_conv = nn.inference(images=x, keep_prob=1.0, is_training=False, ps_device=ps_device, w_device=w_device)
  else:
    y_conv = nn.inference(images=x, keep_prob=1.0, is_training=False, ps_device=ps_device, w_device=w_device)

  label = nn.predict(y_conv)

  with tf.Session() as sess:

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver = tf.train.Saver()
    saver.restore(sess, model_name)

    print("I am self aware!")

    for img_obj in filenames:

      print("Predicting:", img_obj["img"])

      try:

        img, img_np, img_shape_current = image_read(img_obj["img"])

        if(resize_shape):
          
          img_np_x = np.zeros(tf_img_shape)

          prediction_shape = []
          for r_s, i_s in zip(tf_img_shape, img_shape_current):
            if(r_s >= i_s):
              prediction_shape.append("0:" + str(i_s))
            else:
              print("The resize shape is smaller than the current image shape...", img_obj["img"], file=sys.stderr)
              prediction_shape.append("0:")

          assign_img = "img_np_x[" + ",".join(prediction_shape) + "]=np.reshape(img_np, img_shape_current)"
          exec(assign_img)
        else:
          img_np_x = np.reshape(img_np, tf_img_shape)

        label_map = sess.run([label], feed_dict={x: img_np_x.astype(np.float32)})

        if(class_prediction):
          print("Prediction:", class_obj[np.argmax(label_map[0])])
          class_prediction_arr.append({
            "img": img_obj["img"],
            "class": class_obj[np.argmax(label_map[0])],
            "prob": label_map[0].tolist()
            })
        elif(nn.prediction_type() == "image" or nn.prediction_type() == "segmentation"):
          if(resize_shape and resize_prediction):
            label_map = np.array(label_map)
            assign_img = "label_map=label_map[0:1," + ",".join(prediction_shape) + "]"
            exec(assign_img)

          #THE NUMBER OF CHANNELS OF THE OUTPUT ARE GIVEN BY THE NEURAL NET

          Dimension = itk.template(img)[1][1]

          label_map = np.array(label_map[0][0])
          PixelDimension = label_map.shape[-1]
          
          if(PixelDimension < 7):
            if(PixelDimension >= 3 and os.path.splitext(img_obj["out"])[1] not in ['.jpg', '.png']):
              ComponentType = itk.ctype('float')
              PixelType = itk.Vector[ComponentType, PixelDimension]
            elif(PixelDimension == 3):
              PixelType = itk.RGBPixel.UC
              label_map = np.absolute(label_map)
              label_map = np.around(label_map).astype(np.uint16)
            else:
              PixelType = itk.US
              label_map = np.absolute(label_map)
              label_map = np.around(label_map).astype(np.uint16)

            OutputImageType = itk.Image[PixelType, Dimension]
            out_img = OutputImageType.New()

          else:

            ComponentType = itk.ctype('float')
            OutputImageType = itk.VectorImage[ComponentType, Dimension]

            out_img = OutputImageType.New()
            out_img.SetNumberOfComponentsPerPixel(PixelDimension)
            
          size = itk.Size[Dimension]()
          label_map_shape = list(label_map.shape[0:-1])
          label_map_shape.reverse()
          for i, s in enumerate(label_map_shape):
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
          out_img_np.setfield(np.reshape(label_map, out_img_np.shape), out_img_np.dtype)

          print("Writing:", img_obj["out"])
          writer = itk.ImageFileWriter.New(FileName=img_obj["out"], Input=out_img)
          writer.Update()

        elif(nn.prediction_type() == "scalar"):
          prediction_arr.append({
            "img": img_obj["img"],
            "scalar": label_map[0].tolist()
            })
        else:
          print(label_map[0])

      except Exception as e:
        print("Error predicting image:", e, file=sys.stderr)
        print("Continuing...", file=sys.stderr)

    if(class_prediction):
      out_csv = os.path.splitext(out_name)[0] + ".csv"
      print("Writing: ", out_csv)
      with open(out_csv, 'w') as csvfile:
        fieldnames = ['img', 'class', 'prob']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for cp in class_prediction_arr:
          writer.writerow(cp)

    if(prediction_arr):
      print(prediction_arr)

    print("jk, bye")
        