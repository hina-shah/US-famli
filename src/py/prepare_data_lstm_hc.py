#!/usr/bin/env python

from pathlib import Path
import os
import csv
from argparse import ArgumentParser
import logging
import sys
import concurrent.futures
import time
import utils
import json

def getStudyOutputFolder(study, data_folder, out_parent_folder):
    subject_id = study.name
    study_path = study.parent
    study_path_relative = study_path.relative_to(data_folder)
    out_dir = out_parent_folder  / study_path_relative / subject_id
    return out_dir

def main(args):
    data_folder = Path(args.dir)
    out_images_dir = Path(args.out_dir)

    utils.checkDir(out_images_dir, False)
    utils.setupLogFile(out_images_dir, args.debug)

    # read the list of acceptable tags in the ultrasound file
    tags = ['M', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'R15', 'R45',
                    'L15', 'L45', 'L0', 'L1', 'M0', 'M1', 'RTA', 'RTB', 'RTC']
    tag_2d = ['HC']

    info_csv_list = list( data_folder.glob('**/info.csv') )
    logging.info('Found {} studies with an info.csv '.format(len(info_csv_list)))
    print('Found {} studies with an info.csv '.format(len(info_csv_list)))

    bs_db = {}
    counter = 0
    for tag_file in info_csv_list:
        logging.info('--- PROCESSING: {}'.format(tag_file))
        try:
            with open(tag_file) as f:
                csv_reader = csv.DictReader(f)
                bs_list = []
                im_list = []
                for line in csv_reader:
                    if line['tag'] in tags:
                        bs_list.append({'File': line['File'], 'tag': line['tag']})
                    elif line['tag'] in tag_2d and line['type'] == '2d image':
                        im_list.append({'File': line['File']})
                logging.info('BS LIST')
                logging.info(bs_list)
                logging.info('IM LIST')
                logging.info(im_list)
                if len(bs_list) > 0 and len(im_list) > 0:
                    study_name = (tag_file.parent).stem
                    study_det = {}
                    study_det['blindsweeps'] = bs_list
                    study_det['hc_2d'] = im_list
                    bs_db[study_name] = study_det
                    counter+=1
        except (OSError) as e:
            logging.error('Error reading csv file: {}'.format(tag_file))
            return

    try:
        j_f = json.dumps(bs_db, indent=4)
        f = open(str(out_images_dir/'hc_and_blindsweeps.json'),"w")
        f.write(j_f)
        f.close()
    except Exception as e:
        print(e)
        logging.error('Error dumping the json file')
        return

    logging.info('Found {} studies with HC files'.format(counter))
    print('Found {} studies with HC files'.format(counter))
    print('------DONE-----------')


if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with study folders that have an info.csv file generated')
    parser.add_argument('--out_dir', type=str, help='Output directory location.'
                'The directory hierarchy will be copied to this directory')
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    parser.add_argument('--use_threads', action='store_true', help='Use threads to run the code')
    args = parser.parse_args()

    main(args)