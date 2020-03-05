import itk
import argparse
import os
import glob
import numpy as np
import tensorflow as tf
import random
#import matplotlib.pyplot as plt
import sys
import json
import csv
import uuid
from collections import namedtuple
import nrrd3D_2D
import tfRecords_split
from pathlib import Path

def _int64_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def read_image(path, imageDimension):
	if(imageDimension == -1):
		img_read = itk.ImageFileReader.New(FileName=path)
		img_read.Update()
		img = img_read.GetOutput()
	else:
		ImageType = itk.VectorImage[itk.F, args.imageDimension]

		img_read = itk.ImageFileReader[ImageType].New(FileName=path)
		img_read.Update()
		img = img_read.GetOutput()
	number_of_components = img.GetNumberOfComponentsPerPixel()
	return img, number_of_components

def main(args):
	

	print("Input csv", args.csv)
	if(args.enumerate):
		print("Enumerate", args.enumerate)
	print("Output", args.out)

	csv_rows = []

	obj = {}

	with open(args.csv) as csvfile:

		csv_reader = csv.DictReader(csvfile)
		row_keys = csv_reader.fieldnames.copy()

		if("tfRecord" in row_keys):
			row_keys.remove("tfRecord")
		if(not "data_keys" in obj):
			obj["data_keys"] = row_keys

		for row in csv_reader:
			# If we need to slice the images, save 2D slices with the same dimensions
			if args.slice:

				obj["slice"] = True

				slice_imgs = []
				slice_csv_headers = []

				# We check if the path exists, then we read it as an image
				for key in row_keys:
					if os.path.exists(row[key]):
						if not key in slice_csv_headers:
							slice_csv_headers.append(key)
						slice_imgs.append(row[key])

				# We generate an object with the corresponding parameters
				slice_obj = {}
				slice_obj["img"] = slice_imgs
				slice_obj["out_csv_headers"] = slice_csv_headers
				slice_obj["out_csv"] = os.path.join(args.out, "slice.csv")
				slice_obj["out"] = os.path.join(args.out, "slice")

				# We convert the dictionary to a namedtuple, a.k.a, python object, i.e., argparse object
				slice_args = namedtuple("Slice", slice_obj.keys())(*slice_obj.values())
				# Call the main of the script
				nrrd3D_2D.main(slice_args)
				
				# We read the saved data and append the properties of the 3D image to all slices
				with open(slice_obj["out_csv"]) as csvfileslice:
					csv_slice_reader = csv.DictReader(csvfileslice)
					for slice_row in csv_slice_reader:
						for rk in row_keys:
							if(not rk in csv_slice_reader.fieldnames):
								slice_row[rk] = row[rk]
						# We now have a csv_rows list with slices instead of 3D volumes
						csv_rows.append(slice_row)
			else:
				csv_rows.append(row)

	if(not os.path.exists(args.out) or not os.path.isdir(args.out)):
		os.makedirs(args.out)

	if(args.enumerate):
		obj["enumerate"] = args.enumerate
		obj[args.enumerate] = {}
		obj[args.enumerate]["class"] = {}
		#This is the number of classes
		num_class = 0
		print("Enumerate classes...")
		for fobj in csv_rows:
			class_name = fobj[args.enumerate]
			if os.path.exists(class_name):
				#Count number of labels in image
				img_label_read = itk.ImageFileReader.New(FileName=class_name)
				img_label_read.Update()
				img_label = img_label_read.GetOutput()

				label_stats = itk.LabelStatisticsImageFilter[type(img_label),type(img_label)].New()
				label_stats.SetInput(img_label)
				label_stats.SetLabelInput(img_label)
				label_stats.Update()
				
				for class_label in label_stats.GetValidLabelValues():
					if(not class_label in obj[args.enumerate]["class"]):
						obj[args.enumerate]["class"][class_label] = num_class
						num_class += 1

			elif(not class_name in obj[args.enumerate]["class"]):
				obj[args.enumerate]["class"][class_name] = num_class
				num_class += 1
		#Put the total of elements for convenience
		obj[args.enumerate]["num_class"] = num_class

		print(obj[args.enumerate])

	for fobj in csv_rows:
		
		feature = {}
		# This seed is only used when images of different sizes are used
		random_delta_seed = None

		try:

			for key in row_keys:
				if(not key in obj):
					obj[key] = {}

				##If the path exists then it will try to read it as an image
				if(os.path.exists(fobj[key])):
					print("Reading:", fobj[key])
					if(os.path.isdir(fobj[key])):
						# If this is a directory, it will read in all nrrd files,
						# and join together all of them into a single large nrrd file chunk

						# NOTE: assumption is that all of them have the same dimensions
						# NOTE: a length field will also be added to the feature dictionary
						# NOTE: assuming that all images also have the same pixel components
						imglist = list(Path(fobj[key]).glob('*.nrrd'))
						random.shuffle(imglist)
						if len(imglist) == 0:
							print('No images found in this directory, skipping')
							continue
						img_np = None
						sub_im, number_of_components  = read_image(str(imglist[0]), args.imageDimension)
						# Will select/create number of images based on width of the image
						width = sub_im.GetLargestPossibleRegion().GetSize()[0]
						height = sub_im.GetLargestPossibleRegion().GetSize()[1]
						if len(imglist) > args.sequence_length:
							print('Randomly chosing {} images'.format(args.sequence_length))
							imglist = imglist[:args.sequence_length]
						print('Number of images in directory: ',len(imglist))
						
						for img_path in imglist:
							sub_im, number_of_components  = read_image(str(img_path), args.imageDimension)
							sub_im_np = itk.GetArrayViewFromImage(sub_im).astype(float)
							img_np = sub_im_np if img_np is None else np.concatenate([img_np, sub_im_np])
						
						if len(imglist) < args.sequence_length:
							print('Insufficient number of images, padding with zeros:')
							diff = args.sequence_length - len(imglist)
							sub_im_np = np.zeros([diff*height, width] )
							img_np = np.concatenate([img_np, sub_im_np])
						
						print('Shape of the concatenated image: {}'.format(img_np.shape))
					else:
						img, number_of_components = read_image(fobj[key], args.imageDimension)
						img_np = itk.GetArrayViewFromImage(img).astype(float)
					
					if(args.resize):
						obj["resize"] = args.resize
						
						resize_shape = list(args.resize)
						if(number_of_components > 1):
							resize_shape += [number_of_components]
						img_np_x =  np.zeros(resize_shape)

						img_np_assign_shape = []
						# Compute the difference between the shapes
						img_np_shape = img_np.shape
						if(img_np_shape[0] == 1):
						    # If the first component is 1 we remove it. It means that is a 2D image but was saved as 3D
						    img_np_shape = img_np_shape[1:]
						delta_shape = np.array(resize_shape) - np.array(img_np_shape)
						
						if(random_delta_seed is None):
							# We have to use the same seed in the case we are storing multiple images, i.e., per input row
							random_delta_seed = np.random.rand(delta_shape.size)
						random_delta = (random_delta_seed*delta_shape).astype(int)

						# We create the assign operation using the random_delta. We don't want the network to be specific to 
						# the image position
						for s, r in zip(img_np_shape, random_delta):
							img_np_assign_shape.append(str(r) + ":" + str(s+r))

						assign_img = "img_np_x[" + ",".join(img_np_assign_shape) + "] = img_np"
						exec(assign_img)
						img_np = img_np_x
					
					feature[key] =  _float_feature(img_np.reshape(-1).tolist())

					# Put the shape of the image in the json object if it does not exists. This is done for global information
					if args.lstmMode and os.path.isdir(fobj[key]):
						img_shape = [args.sequence_length, width, height]
					else:
						img_shape = list(img_np.shape)
					if(img_shape[0] == 1):
						# If the first component is 1 we remove it. It means that is a 2D image but was saved as 3D
						img_shape = img_shape[1:]

					# This is the number of channels, if the number of components is 1, it is not included in the image shape
					# If it has more than one component, it is included in the shape, that's why we have to add the 1
					if(number_of_components == 1):
						img_shape = img_shape + [1]

					# This boolean will mark that this key is from a directory
					# And also add num_images into the data keys
					if args.lstmMode and os.path.isdir(fobj[key]) and not "dirFeature" in obj[key]:
						obj[key]["dirFeature"] = True

					if(not "shape" in obj[key]):
						print("Shape", key, img_shape)
						obj[key]["shape"] = img_shape
					else:
						if not np.all(np.equal(obj[key]["shape"] , img_shape)):
							print(fobj[key], file=sys.stderr)
							print("The images in your training set do not have the same dimensions!")
							raise "The images in your training set do not have the same dimensions!"

					if(not "max" in obj[key]):
						obj[key]["max"] = np.max(img_np)
					else:
						obj[key]["max"] = max(obj[key]["max"], np.max(img_np))

					if(not "min" in obj[key]):
						obj[key]["min"] = np.min(img_np)
					else:
						obj[key]["min"] = min(obj[key]["min"], np.min(img_np))

					if(not "type" in obj[key]):
						obj[key]["type"] = "tf.float32"
					
				elif(args.enumerate and key == args.enumerate):
					# If its an enumeration, it will save the class enumeration as int. If is a label map, it never reached this step
					# because it was read as an image. 
					class_name = fobj[key]
					class_number = int(obj[args.enumerate]["class"][class_name])

					feature[key] = _int64_feature(class_number)

					if(not "shape" in obj[key]):
						obj[key]["shape"] = [1]

					if(not "type" in obj[key]):
						obj[key]["type"] = "tf.int64"

				else:
					try:
						# If the previous failed, try converting it to float
						feature[key] = _float_feature(np.array([float(fobj[key])]).tolist())

						if(not "shape" in obj[key]):
							obj[key]["shape"] = [1]

						if(not "type" in obj[key]):
							obj[key]["type"] = "tf.float32"
					except Exception as e:
						# If it fails the try saving it as a bytes feature
						# encode the string
						feature[key] = _bytes_feature(obj[key].encode())

						if(not "shape" in obj[key]):
							obj[key]["shape"] = [1]

						if(not "type" in obj[key]):
							obj[key]["type"] = "tf.string"
		
			if("tfRecord" in fobj):
				record_path = fobj["tfRecord"]
			else:
				record_path = os.path.join(args.out, str(uuid.uuid4()) + ".tfrecord")

			fobj["tfRecord"] = record_path

			writer = tf.io.TFRecordWriter(record_path)
			example = tf.train.Example(features=tf.train.Features(feature=feature))

			print("Writing record", fobj)

			writer.write(example.SerializeToString())
			writer.close()

		except Exception as e:
			print("Error converting to tfRecord: \n ", obj, e, file=sys.stderr)
			print("\n")
			print("I'll keep on going...")
	
	obj['tfrecords'] = os.path.basename(args.out.rstrip(os.sep))
	
	outjson = args.out.rstrip(os.sep) + ".json"
	print("Writing:", outjson)

	print(obj)
	with open(outjson, "w") as f:
		f.write(json.dumps(obj))
	
	outcsv = None
	if(len(csv_rows) > 0):
		outcsv = os.path.splitext(args.csv)[0] + "_tfRecords.csv"
		print("Writing:", outcsv)
		csv_headers = list(csv_rows[0].keys())
		print(csv_headers)
		with open(outcsv, "w") as f:
			writer = csv.DictWriter(f, fieldnames=csv_headers)
			writer.writeheader()
			for row in csv_rows:
				writer.writerow(row)

	if(args.split > 0 and args.split < 1):
		# We generate an object with the corresponding parameters
		split_obj = {}
		split_obj["json"] = outjson
		split_obj["split"] = args.split
		split_obj["csv"] = outcsv

		# We convert the dictionary to a namedtuple, a.k.a, python object, i.e., argparse object
		split_args = namedtuple("Split", split_obj.keys())(*split_obj.values())
		# Call the main of the script
		tfRecords_split.main(split_args)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Writes data to tfrecords format using a CSV file description as input and creates a JSON file with a description. The column headers are used as keys to store in the tfRecord. a row may contain image filenames, categories or other information to save in tfRecords format. If an image column is found, the maximum pixel value for each image is calulated and written into the json file. Additionally, it will save a new csv file indicating which tfRecord corresponds to the row.')
	
	parser.add_argument('--csv', type=str, help='CSV file with dataset information,', required=True)
	parser.add_argument('--enumerate', type=str, default=None, help='Column name in CSV. If you are storing a label or category to perform a classification task. If it is an image, it will read the FIRST image in your csv and extract the existing labels.')
	parser.add_argument('--slice', type=bool, default=False, help="If it is a 3D image, saves slices in all the major axis and the stores them as tfRecords")
	parser.add_argument('--resize', nargs="+", type=int, default=None, help='Resize images to store as tfRecord. The resize parameter must be equal or larger than the largest image in the dataset. Do not include channels and flip axes, i.e, z y x or y x for 3D, 2D images respectively')
	parser.add_argument('--out', type=str, default="./out", help="Output directory")
	parser.add_argument('--split', type=float, default=0, help="Split the data for evaluation. [0-1], 0=no split")
	parser.add_argument('--imageDimension', type=int, default=-1, help="Set image dimension, by default it will try to guess it but you can set this here. Or use it when a type is not wrapped. Ex. RGB double, it will read the image as a float vector image")
	parser.add_argument('--lstmMode', type=bool, default=False, help="LSTM mode ignores the requirement that all feature should have the same size")
	parser.add_argument('--sequence_length', type=int, default=-1, help="Specifies number of frames in each video for lstm mode. This is required when LSTM mode is used.")
	args = parser.parse_args()

	main(args)
