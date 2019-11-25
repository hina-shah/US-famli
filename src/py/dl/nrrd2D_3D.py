import itk
import argparse
import os
import numpy as np
import uuid
import sys
import csv
import json


def main(args):

	img_filenames = []

	with open(args.csv) as csvfile:

		csv_reader = csv.DictReader(csvfile)
		row_keys = csv_reader.fieldnames.copy()

		for row in csv_reader:
			img_filenames.append(row[args.img_col])

	if(len(img_filenames) > 0):
		
		InputPixelType = itk.F
		InputImageType = itk.Image[InputPixelType, 2]

		img_read = itk.ImageFileReader[InputImageType].New(FileName=img_filenames[0])
		img_read.Update()
		img = img_read.GetOutput()

		PixelType = itk.template(img)[1][0]
		OutputImageType = itk.Image[PixelType, 2]

		PixelType = itk.template(img)[1][0]
		OutputImageType = itk.Image[PixelType, 3]
		InputImageType = itk.Image[PixelType, 2]

		tileFilter = itk.TileImageFilter[InputImageType, OutputImageType].New()

		layout = [1, 1, 0]
		tileFilter.SetLayout(layout)

		imgs_arr_description = []

		for i, filename in enumerate(img_filenames):
			print("Reading:", filename)
			img_read = itk.ImageFileReader[InputImageType].New(FileName=filename)
			img_read.Update()
			img = img_read.GetOutput()

			tileFilter.SetInput(i, img)

			img_obj_description = {}
			img_obj_description["image"] = {}
			img_obj_description["image"]["region"] = {}
			img_obj_description["image"]["region"]["size"] = np.array(img.GetLargestPossibleRegion().GetSize()).tolist()
			img_obj_description["image"]["region"]["index"] = np.array(img.GetLargestPossibleRegion().GetIndex()).tolist()
			img_obj_description["image"]["spacing"] = np.array(img.GetSpacing()).tolist()
			img_obj_description["image"]["origin"] = np.array(img.GetOrigin()).tolist() 
			img_obj_description["image"]["direction"] = itk.GetArrayFromVnlMatrix(img.GetDirection().GetVnlMatrix().as_matrix()).tolist()
			img_obj_description["img_filename"] = os.path.basename(filename)

			imgs_arr_description.append(img_obj_description)

		defaultValue = 0
		tileFilter.SetDefaultPixelValue(defaultValue)
		tileFilter.Update()

		writer = itk.ImageFileWriter[OutputImageType].New()
		writer.SetFileName(args.out)
		writer.SetInput(tileFilter.GetOutput())
		writer.Update()

		with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
			f.write(json.dumps(imgs_arr_description))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', type=str, help='Input CSV file with images', required=True)
	parser.add_argument('--img_col', type=str, help='Column name in the CSV file that contains the images', default="image")
	parser.add_argument('--out', type=str, help='Output filename', default="out.nrrd")

	args = parser.parse_args()

	main(args)