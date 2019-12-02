import itk
import argparse
import os
import numpy as np
import uuid
import sys
import csv
import json

def main(args):
	
	if(args.json):
		imgs_arr = []

		for img_index, img_filename in enumerate(args.img):
			img_read = itk.ImageFileReader.New(FileName=img_filename)
			img_read.Update()
			img = img_read.GetOutput()
			imgs_arr.append(img)

		description_arr = []
		with open(args.json, "r") as jsf:
			description_arr = json.load(jsf)

		for img in imgs_arr:

			img_np = itk.GetArrayViewFromImage(img)

			for img_index in range(img_np.shape[0]):

				description = description_arr[img_index]

				RegionType = itk.ImageRegion[2]
				region = RegionType()
				region.SetIndex(description["image"]["region"]["index"])
				region.SetSize(description["image"]["region"]["size"])

				PixelType = itk.template(img)[1][0]
				OutputImageType = itk.Image[PixelType, 2]
				
				out_img = OutputImageType.New()
				out_img.SetRegions(region)
				out_img.SetOrigin(description["image"]["origin"])
				out_img.SetSpacing(description["image"]["spacing"])
				
				np_dir_vnl = itk.GetVnlMatrixFromArray(np.array(description["image"]["direction"]))
				direction = out_img.GetDirection()
				direction.GetVnlMatrix().copy_in(np_dir_vnl.data_block())

				out_img.Allocate()

				out_img_np = itk.GetArrayViewFromImage(out_img)

				out_img_np.setfield(img_np[img_index], out_img_np.dtype)

				if(not os.path.exists(args.out) or not os.path.isdir(args.out)):
					os.makedirs(args.out)

				out_filename = os.path.join(args.out, description["img_filename"])
				print("Writing:", out_filename)
				writer = itk.ImageFileWriter.New(FileName=out_filename, Input=out_img)
				writer.Update()

	else:
		imgs_arr = []
		imgs_arr_description = []
		
		csv_headers = []
		if args.out_csv_headers:
			csv_headers = args.out_csv_headers
		
		for img_index, img_filename in enumerate(args.img):
			img_read = itk.ImageFileReader.New(FileName=img_filename)
			img_read.Update()
			img = img_read.GetOutput()
			imgs_arr.append(img)

			if args.out_csv_headers is None:
				csv_headers.append(img_index)

			img_obj_description = {}
			img_obj_description["image"] = {}
			img_obj_description["image"]["region"] = {}
			img_obj_description["image"]["region"]["size"] = np.array(img.GetLargestPossibleRegion().GetSize()).tolist()
			img_obj_description["image"]["region"]["index"] = np.array(img.GetLargestPossibleRegion().GetIndex()).tolist()
			img_obj_description["image"]["spacing"] = np.array(img.GetSpacing()).tolist()
			img_obj_description["image"]["origin"] = np.array(img.GetOrigin()).tolist() 
			img_obj_description["image"]["direction"] = itk.GetArrayFromVnlMatrix(img.GetDirection().GetVnlMatrix().as_matrix()).tolist()
			img_obj_description["img_index"] = img_index
			img_obj_description["img_filename"] = img_filename
			img_obj_description["slice"] = {}

			imgs_arr_description.append(img_obj_description)

		if len(args.img) != len(csv_headers):
			print("The csv headers  provided do not have the same length as the number of images provided", file=sys.stderr)
			sys.exit(1)
		
		if len(imgs_arr) == 0:
			print("You need to provide at least one image!", file=sys.stderr)
			sys.exit(1)
						

		print(imgs_arr_description)
		img_np = itk.GetArrayViewFromImage(imgs_arr[0])
		img_shape_max = np.max(img_np.shape)
		out_size = np.array([img_shape_max, img_shape_max]).tolist()
		
		PixelType = itk.template(imgs_arr[0])[1][0]
		OutputImageType = itk.Image[PixelType, 2]

		out_csv_rows = []

		if imgs_arr[0].GetImageDimension() == 3:
			img_shape = img_np.shape
			for dim in range(imgs_arr[0].GetImageDimension()):
				
				imgs_arr_description[img_index]["slice"][dim] = []

				for sln in range(img_shape[dim]):

					start = [0,0,0]
					end = [None,None,None]
					start[dim] = sln
					end[dim] = sln + 1
					uuid_out_name = str(uuid.uuid4())

					out_obj_csv = {}

					for img_index, img in enumerate(imgs_arr):

						img_np = itk.GetArrayViewFromImage(img)
						
						img_np_2d = np.squeeze(img_np[start[0]:end[0],start[1]:end[1],start[2]:end[2]], axis=dim)
						size_np = img_np_2d.shape

						img_np_2d_max_size = np.zeros(out_size)
						img_np_2d_max_size[0:size_np[0],0:size_np[1]] = img_np_2d

						index = itk.Index[2]()
						index.Fill(0)

						RegionType = itk.ImageRegion[2]
						region = RegionType()
						region.SetIndex(index)
						region.SetSize(out_size)

						out_img = OutputImageType.New()
						out_img.SetRegions(region)
						out_img.SetOrigin(np.delete(img.GetOrigin(),dim).tolist())
						out_img.SetSpacing(np.delete(img.GetSpacing(),dim).tolist())
						out_img.Allocate()

						out_img_np = itk.GetArrayViewFromImage(out_img)

						out_img_np.setfield(img_np_2d_max_size, out_img_np.dtype)
			
						out_dir = os.path.join(args.out, str(csv_headers[img_index]))

						if(not os.path.exists(out_dir) or not os.path.isdir(out_dir)):
							os.makedirs(out_dir)

						out_filename = os.path.join(out_dir, uuid_out_name + ".nrrd")
						print("Writing:", out_filename)
						writer = itk.ImageFileWriter.New(FileName=out_filename, Input=out_img)
						writer.Update()
						
						out_obj_csv[csv_headers[img_index]] = out_filename

						imgs_arr_description[img_index]["slice"][dim].append(out_filename)

					out_csv_rows.append(out_obj_csv)

			with open(args.out_csv, "w") as f:
				writer = csv.DictWriter(f, fieldnames=csv_headers)
				writer.writeheader()
				for row in out_csv_rows:
					writer.writerow(row)

			if(args.json_desc):
				with open(args.json_desc, "w") as f:
					f.write(json.dumps(imgs_arr_description))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', nargs="+", type=str, help='Input image', required=True)
	parser.add_argument('--json', type=str, help='Input JSON file created by nrrd2D_3D', default=None)
	parser.add_argument('--out_csv', type=str, help='Output csv filename', default="out.csv")
	parser.add_argument('--out_csv_headers', nargs="+", type=str, help='Output csv headers', default=None)
	parser.add_argument('--out', type=str, help='Output directory', default="./")
	parser.add_argument('--json_desc', type=str, help='Output json description for each image', default=None)

	args = parser.parse_args()

	main(args)