import json
import argparse
import numpy as np
import math
import sys

def circumference_ellipse(radius):
	if(len(radius) < 2):
		print("Radius must be an array with two values", file=sys.stderr)
		return 0
	a = radius[0]
	b = radius[1]
	if a < 0 or b < 0:
		print('Values to circumference_ellipse must be greater than 0', file=sys.stderr)
		return 0

	return math.pi * ( 3*(a + b) - math.sqrt( (3*a + b) * (a + 3*b) ) )

def circumference_circle(radius):
	if(len(radius) < 1):
		print("Radius must be an array one values", file=sys.stderr)
		return 0

	a = radius[0]
	if a < 0:
		print('in Circle circumference, radius must be greathre than 0', file = sys.stderr)
		return 0

	return 2 * math.pi * a 

def distance_points(p1, p2):
	return np.linalg.norm(np.array(p1) - np.array(p2))

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Computes the circumference or length of line based on the scale of the image. This is only good for the dataset_a US-FAMLI. Image are in JPG and the scale is drawn from the image', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--scale', type=float, help='Scale', default=1.0)
	parser.add_argument('--gt', type=str, help='Scale', default="1.0cm")
	parser.add_argument('--json', help='JSON file with fit description')

	args = parser.parse_args()

	with open(args.json, "r") as f:
		fit_description = json.load(f)

		scale = args.scale/400
		if("radius" in fit_description):

			if(len(fit_description["radius"]) > 1):
				circumference = circumference_ellipse(fit_description["radius"])
				output=[args.json.replace(".json",".nrrd"),str(circumference*scale),str(2.0*np.min(fit_description["radius"])*scale),str(args.scale),args.gt.replace("cm","")]
				print(",".join(output))
			elif(len(fit_description["radius"]) == 1):
				circumference = circumference_circle(fit_description["radius"])
				output=[args.json.replace(".json",".nrrd"),str(circumference*scale),str(args.scale),args.gt.replace("cm","")]
				print(",".join(output))
		elif("min" in fit_description and "max" in fit_description):
			distance = distance_points(fit_description["min"], fit_description["max"])
			output=[args.json.replace(".json",".nrrd"),str(distance*scale),str(args.scale),args.gt.replace("cm","")]
			print(",".join(output))

