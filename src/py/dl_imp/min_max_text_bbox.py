import itk
import argparse
import os
import numpy as np
import sys
import json

def main(args):

	img_read = itk.ImageFileReader.New(FileName=args.img)
	img_read.Update()
	img = img_read.GetOutput()

	
	with open(args.json_m, "r") as jsm:
		min_max_obj = json.load(jsm)

		Dimension = len(img.GetLargestPossibleRegion().GetSize())
		PixelType = itk.ctype('unsigned short')
		OutputImageType = itk.Image[PixelType, Dimension]

		out_img = OutputImageType.New()

		out_img.SetRegions(img.GetLargestPossibleRegion())
		out_img.SetDirection(img.GetDirection())
		out_img.SetOrigin(img.GetOrigin())
		out_img.SetSpacing(img.GetSpacing())
		out_img.Allocate()
		out_img.FillBuffer(0)

		out_img_np = itk.GetArrayViewFromImage(out_img)
		out_img_np[int(min_max_obj["min"][1] - args.pady):int(min_max_obj["max"][1] + args.pady),int(min_max_obj["min"][0] - args.padx):int(min_max_obj["max"][0] + args.padx)] = 1

		print("Writing:", args.out)
		writer = itk.ImageFileWriter[OutputImageType].New(FileName=args.out, Input=out_img)
		writer.Update()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', type=str, help='Input image', required=True)
	parser.add_argument('--out', type=str, help='Output image', default="out.nrrd")
	parser.add_argument('--json_m', type=str, help='Output json description for each image', required=True)
	parser.add_argument('--padx', type=int, help='Pad pixels for x', default=30)
	parser.add_argument('--pady', type=int, help='Pad pixels for y', default=30)

	args = parser.parse_args()

	main(args)