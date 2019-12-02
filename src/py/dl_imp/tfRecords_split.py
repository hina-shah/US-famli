import argparse
import os
import glob
import numpy as np
import sys
import json
import csv

def main(args):
	
	json_filename = args.json
	data_description = {}

	with open(json_filename, "r") as f:
		data_description = json.load(f)

	tfrecords_arr = []
	tfrecords_dir = os.path.join(os.path.dirname(json_filename), data_description["tfrecords"], '**/*.tfrecord')
	for tfr in glob.iglob(tfrecords_dir, recursive=True):
		tfrecords_arr.append(tfr)

	tfrecords_arr = np.array(tfrecords_arr)
	tfrecords_arr = tfrecords_arr[np.random.permutation(tfrecords_arr.size)]

	train_samples = int(tfrecords_arr.size - tfrecords_arr.size*args.split)

	tfrecords_train_dir = os.path.join(os.path.dirname(json_filename), data_description["tfrecords"] + "_split_train")
	data_description_train  = {}
	data_description_train.update(data_description)
	data_description_train["tfrecords"] = os.path.basename(tfrecords_train_dir)

	if(not os.path.exists(tfrecords_train_dir)):
		os.makedirs(tfrecords_train_dir)

	for tfr in tfrecords_arr[0:train_samples]:
		try:
			dest = os.path.join(tfrecords_train_dir, os.path.basename(tfr))
			print("Creating soft link: ", dest)
			os.symlink(tfr, dest)
		except Exception as e:
			print("Error making symlink", e, file=sys.stderr)

	tfrecords_train_json = tfrecords_train_dir.rstrip(os.sep) + ".json"
	with open(tfrecords_train_json, "w") as f:
		f.write(json.dumps(data_description_train))

	tfrecords_eval_dir = os.path.join(os.path.dirname(json_filename), data_description["tfrecords"] + "_split_eval")
	data_description_eval  = {}
	data_description_eval.update(data_description)
	data_description_eval["tfrecords"] = os.path.basename(tfrecords_eval_dir)

	if(not os.path.exists(tfrecords_eval_dir)):
		os.makedirs(tfrecords_eval_dir)

	for tfr in tfrecords_arr[train_samples:]:
		try:
			dest = os.path.join(tfrecords_eval_dir, os.path.basename(tfr))
			print("Creating soft link: ", dest)
			os.symlink(tfr, dest)
		except Exception as e:
			print("Error making symlink", e, file=sys.stderr)
	
	tfrecords_eval_json = tfrecords_eval_dir.rstrip(os.sep) + ".json"
	with open(tfrecords_eval_json, "w") as f:
		f.write(json.dumps(data_description_eval))

	if(args.csv):
		with open(args.csv) as csvfile:

			csv_reader = csv.DictReader(csvfile)
			row_keys = csv_reader.fieldnames.copy()

			if("tfRecord" in row_keys):
				tf_train_array = []
				tf_eval_array = []
				for row in csv_reader:
					tfrecord_filename = os.path.join(tfrecords_train_dir, os.path.basename(row["tfRecord"]))
					if(os.path.isfile(tfrecord_filename)):
						tf_train_array.append(row)

					tfrecord_filename = os.path.join(tfrecords_eval_dir, os.path.basename(row["tfRecord"]))
					if(os.path.isfile(tfrecord_filename)):
						tf_eval_array.append(row)

				if(len(tf_train_array) > 0):
					outcsv = os.path.splitext(args.csv)[0] + "_train.csv"
					with open(outcsv, "w") as f:
						writer = csv.DictWriter(f, fieldnames=row_keys)
						writer.writeheader()
						for row in tf_train_array:
							writer.writerow(row)

				if(len(tf_eval_array) > 0):
					outcsv = os.path.splitext(args.csv)[0] + "_eval.csv"
					with open(outcsv, "w") as f:
						writer = csv.DictWriter(f, fieldnames=row_keys)
						writer.writeheader()
						for row in tf_eval_array:
							writer.writerow(row)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Splits the data by creating soft links and new folders for training and evaluation purposes')
	
	parser.add_argument('--json', type=str, help='JSON file created by tfRecords.py', required=True)
	parser.add_argument('--split', type=float, default=0, help="Split the data for evaluation. [0-1], 0=no split")
	parser.add_argument('--csv', type=str, default=None, help="If provided, generates the corresponding CSV files for the splitted dataset")

	args = parser.parse_args()

	main(args)
