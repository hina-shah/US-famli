import argparse
import os
import numpy as np
import json

def sort_text(text_obj, ypixtolerance=10):

	lines_arr = []
	for tobj in text_obj:
		y = tobj["y"]
		found = False
		for la in lines_arr:
			if(la["min"] <= y and y <= la["max"]):
				la["text_obj"].append(tobj)
				found = True
		if(not found):
			lines_arr.append({"min": y - ypixtolerance, "max": y + ypixtolerance, "text_obj": [tobj]})

	lines_arr = sorted(lines_arr, key=lambda k: k["min"])

	for la in lines_arr:
		la["text_obj"] = sorted(la["text_obj"], key=lambda k: k["x"])

	text_obj_a = []

	for la in lines_arr:
		text_obj_a.extend(la["text_obj"])

	return text_obj_a


def main(args):
	with open(args.json, "r") as f:
		text_desc = json.load(f)

		text_description=[args.json]
		text_obj = []
		for text in text_desc:
			if text["bounds"] and text["bounds"][0] and text["bounds"][0]["y"]:
				xavg = 0
				yavg = 0
				for tb in text["bounds"]:
					xavg += tb["x"]
					yavg += tb["y"]
				yavg /= len(text["bounds"])
				xavg /= len(text["bounds"])
				
				if args.miny < yavg and yavg < args.maxy and args.minx < xavg and xavg < args.maxx:
					t_obj = {"y": yavg, "x": xavg, "description": text["description"]}
					text_obj.append(t_obj)

		text_obj = sort_text(text_obj)

		for to in text_obj:
			text_description.append(to["description"])

		if len(text_description) > 1:
			print(','.join(text_description))

		if(args.out_m):
			x_coord = [to["x"] for to in text_obj]
			y_coord = [to["y"] for to in text_obj]

			if(len(x_coord) and len(y_coord)):

				min_max_obj = {}
				min_max_obj["min"] = [np.min(x_coord), np.min(y_coord)]
				min_max_obj["max"] = [np.max(x_coord), np.max(y_coord)]

				with open(args.out_m, "w") as f:
					f.write(json.dumps(min_max_obj))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--json', type=str, help='JSON filename produced by detect_text.py')
	
	parser.add_argument('--minx', type=int, help='x coordinate min', default=40)
	parser.add_argument('--maxx', type=int, help='x coordinate max', default=800)
	parser.add_argument('--miny', type=int, help='x coordinate min', default=405)
	parser.add_argument('--maxy', type=int, help='x coordinate max', default=420)
	parser.add_argument('--join', type=str, help='join character', default=',')
	parser.add_argument('--out_m', type=str, help='Output the minimum and maximum coords of the text found in a JSON file', default=None)

	args = parser.parse_args()

	main(args)