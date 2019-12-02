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
	
	with open(args.json, "r") as js:

		text_desc = json.load(js)
		
		for i in range(args.start, len(text_desc)):
			text = text_desc[i]
			if "bounds" in text:

				x_coord = [to["x"] for to in text["bounds"]]
				y_coord = [to["y"] for to in text["bounds"]]

				min_max_obj = {}
				min_max_obj["min"] = [np.min(x_coord), np.min(y_coord)]
				min_max_obj["max"] = [np.max(x_coord), np.max(y_coord)]
				out_img_np[int(min_max_obj["min"][1] - args.pady):int(min_max_obj["max"][1] + args.pady),int(min_max_obj["min"][0] - args.padx):int(min_max_obj["max"][0] + args.padx)] = 1


	print("Writing:", args.out)
	writer = itk.ImageFileWriter[OutputImageType].New(FileName=args.out, Input=out_img)
	writer.Update()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Outputs a label image where text is found')
	parser.add_argument('--img', type=str, help='Input image', required=True)
	parser.add_argument('--out', type=str, help='Output image', default="out.nrrd")
	parser.add_argument('--json', type=str, help='JSON filename produced by detect_text.py', required=True)
	parser.add_argument('--start', type=int, help='Start at this position to create the label map. The first position usually contains all the text found and the bounding box overlaps the whole image, i.e, not good.', default=1)
	parser.add_argument('--padx', type=int, help='Pad pixels for x', default=1)
	parser.add_argument('--pady', type=int, help='Pad pixels for y', default=1)

	args = parser.parse_args()

	main(args)