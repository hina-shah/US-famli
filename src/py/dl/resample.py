import itk
import numpy as np
import argparse
import os
import glob
import sys
import csv

def Resample(img_filename, args):

	output_size = args.size 
	fit_spacing = args.fit_spacing
	iso_spacing = args.iso_spacing
	img_dimension = args.dimension
	pixel_dimension = args.pixel_dimension

	if(pixel_dimension == 1):
		zeroPixel = 0
		VectorImageType = itk.Image[itk.F, img_dimension]
	else:
		zeroPixel = np.zeros(pixel_dimension)
		if(args.rgb):
			if(pixel_dimension == 3):
				PixelType = itk.RGBPixel[itk.UC]
			else:
				PixelType = itk.itk.RGBAPixel[itk.UC]
		else:
			PixelType = itk.Vector[itk.F, pixel_dimension]
		VectorImageType = itk.Image[PixelType, img_dimension]

	print("Reading:", img_filename)
	img_read = itk.ImageFileReader[VectorImageType].New(FileName=img_filename)
	img_read.Update()
	img = img_read.GetOutput()

	if args.linear:
		InterpolatorType = itk.LinearInterpolateImageFunction[VectorImageType, itk.D]
	else:
		InterpolatorType = itk.NearestNeighborInterpolateImageFunction[VectorImageType, itk.D]

	ResampleType = itk.ResampleImageFilter[VectorImageType, VectorImageType]

	spacing = img.GetSpacing()
	region = img.GetLargestPossibleRegion()
	size = region.GetSize()

	output_size = [si if o_si == -1 else o_si for si, o_si in zip(size, output_size)]
	print(output_size)

	if(fit_spacing):
		output_spacing = [sp*si/o_si for sp, si, o_si in zip(spacing, size, output_size)]
	else:
		output_spacing = spacing

	if(iso_spacing):
		max_spacing = np.max(output_spacing)
		output_spacing = np.ones_like(output_spacing)*max_spacing

	if(args.spacing is not None):
		output_spacing = args.spacing
	
	resampleImageFilter = ResampleType.New()
	interpolator = InterpolatorType.New()

	resampleImageFilter.SetDefaultPixelValue(zeroPixel)
	resampleImageFilter.SetOutputSpacing(output_spacing)
	resampleImageFilter.SetOutputOrigin(img.GetOrigin())

	resampleImageFilter.SetInterpolator(interpolator)
	resampleImageFilter.SetSize(output_size)
	resampleImageFilter.SetInput(img)
	resampleImageFilter.Update()

	return resampleImageFilter.GetOutput()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Resample an image', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	in_group = parser.add_mutually_exclusive_group(required=True)

	in_group.add_argument('--img', type=str, help='image to resample')
	in_group.add_argument('--dir', type=str, help='Directory with image to resample')
	in_group.add_argument('--csv', type=str, help='CSV file with column img with paths to images to resample')

	parser.add_argument('--csv_column', type=str, default='image', help='CSV column name (Only used if flag csv is used)')
	parser.add_argument('--csv_root_path', type=str, default='', help='Replaces a root path directory to empty, this is use to recreate a directory structure in the output directory, otherwise, the output name will be the name in the csv (only if csv flag is used)')
	parser.add_argument('--size', nargs="+", type=int, help='Output size, -1 to leave unchanged', default=None)
	parser.add_argument('--linear', type=bool, help='Use linear interpolation.', default=False)
	parser.add_argument('--spacing', nargs="+", type=float, default=None, help='Use a pre defined spacing')
	parser.add_argument('--fit_spacing', type=bool, help='Fit spacing to output', default=False)
	parser.add_argument('--iso_spacing', type=bool, help='Same spacing for resampled output', default=False)
	parser.add_argument('--dimension', type=int, help='Image dimension', default=2)
	parser.add_argument('--pixel_dimension', type=int, help='Pixel dimension', default=1)
	parser.add_argument('--rgb', type=bool, help='Use RGB type pixel', default=False)
	parser.add_argument('--ow', type=int, help='Overwrite', default=1)
	parser.add_argument('--out', type=str, help='Output image/directory', default="./out.nrrd")
	parser.add_argument('--out_ext', type=str, help='Output extension type', default=None)

	args = parser.parse_args()

	filenames = []
	if(args.img):
		fobj = {}
		fobj["img"] = args.img
		fobj["out"] = args.out
		filenames.append(fobj)
	elif(args.dir):
		out_dir = args.out
		normpath = os.path.normpath("/".join([args.dir, '**', '*']))
		for img in glob.iglob(normpath, recursive=True):
			if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png"]]:
				fobj = {}
				fobj["img"] = img
				fobj["out"] = os.path.normpath(out_dir + "/" + img.replace(args.dir, ''))
				if args.out_ext is not None:
					fobj["out"] = os.path.splitext(fobj["out"])[0] + args.out_ext
				if not os.path.exists(os.path.dirname(fobj["out"])):
					os.makedirs(os.path.dirname(fobj["out"]))
				if not os.path.exists(fobj["out"]) or args.ow:
					filenames.append(fobj)
	elif(args.csv):
		replace_dir_name = args.csv_root_path
		with open(args.csv) as csvfile:
			csv_reader = csv.DictReader(csvfile)
			for row in csv_reader:
				fobj = {}
				fobj["img"] = row[args.csv_column]
				fobj["out"] = row[args.csv_column].replace(args.csv_root_path, args.out)
				if args.out_ext is not None:
					fobj["out"] = os.path.splitext(fobj["out"])[0] + args.out_ext
				if not os.path.exists(os.path.dirname(fobj["out"])):
					os.makedirs(os.path.dirname(fobj["out"]))
				if not os.path.exists(fobj["out"]) or args.ow:
					filenames.append(fobj)
	else:
		raise "Set img or dir to resample!"

	if(args.rgb):
		if(args.pixel_dimension == 3):
			print("Using: RGB type pixel with unsigned char")
		elif(args.pixel_dimension == 4):
			print("Using: RGBA type pixel with unsigned char")
		else:
			print("WARNING: Pixel size not supported!")

	for fobj in filenames:

		try:

			if args.size is not None:
				img = Resample(fobj["img"], args)
			else:
				img_dimension = args.dimension
				pixel_dimension = args.pixel_dimension

				if(pixel_dimension == 1):
					VectorImageType = itk.Image[itk.F, img_dimension]
				else:
					if(args.rgb):
						if(pixel_dimension == 3):
							PixelType = itk.RGBPixel[itk.UC]
						else:
							PixelType = itk.itk.RGBAPixel[itk.UC]
					else:
						PixelType = itk.Vector[itk.F, pixel_dimension]
					VectorImageType = itk.Image[PixelType, img_dimension]

				print("Reading:", fobj["img"])
				img_read = itk.ImageFileReader[VectorImageType].New(FileName=fobj["img"])
				img_read.Update()
				img = img_read.GetOutput()

			print("Writing:", fobj["out"])
			WriterType = itk.ImageFileWriter[img]
			writer = WriterType.New()
			writer.SetInput(img)
			writer.SetFileName(fobj["out"])
			writer.UseCompressionOn()
			writer.Update()
		except Exception as e:
			print(e, file=sys.stderr)
