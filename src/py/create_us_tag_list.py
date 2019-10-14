#!/usr/bin/env python

from pathlib import Path
import os
import csv
from argparse import ArgumentParser
from pprint import pprint
import logging
import sys
import preprocessus
import extracttagtext

def main(args):
    data_folder = Path(args.dir)
    out_images_dir = Path(args.out_dir)

    preprocessus.checkDir(out_images_dir, False)    
    logging.basicConfig(level=logging.INFO, filename=str(out_images_dir / 'log.txt'))

    studies = []
    for dirname, dirnames, __ in os.walk(str(data_folder)):
        if len(dirnames) == 0:
            studies.append(Path(dirname))
            
    logging.info('Found {} studies '.format(len(studies)))
    # subjects = ["UNC-0414-1_20190830_111550", 
    #             "UNC-0414-2_20190924_085140",
    #             "UNC-0366-3_20190927_093424",
    #             "UNC-0394-3_20191001_113111",
    #             "UNC-0418-1_20190904_090711",
    #             "UNC-0447-1_20190927_114657"]
    
    rescale_size = [800,600]
    tag_list_file = 'us_tags.txt'
    
    # Approximate bounding box of where the tag is written acoording to the 
    # us model
    tag_bounding_box = { 'V830':[[40,75], [255,190]],
                        'LOGIQe':  [[0,55], [200,160]]}

    # list of ultrasound image types whose tags we do not care about right now.
    non_tag_us = ['Unknown', 'ge kretz image', 'Secondary capture image report',
                    'Comprehensive SR', '3D Dicom Volume']

    # read the list of acceptable tags in the ultrasound file
    tag_list = None
    try:
        with open(Path(__file__).parent / tag_list_file) as f:
            tag_list = f.read().splitlines()
    except:
        logging.warning('ERROR READING THE TAG FILE')
        tag_list = ['M', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'R15', 'R45', 
                    'L15', 'L45', 'L0', 'L1', 'M0', 'M1', 'RTA', 'RTB', 'RTC']

    tag_statistic = dict.fromkeys(tag_list, 0)
    tag_statistic['Unknown'] = 0
    tag_statistic['Undecided'] = 0
    tag_statistic['No tag'] = 0

    # Start processing all
    preprocessus.checkDir(out_images_dir, False)
    for study in studies:
        subject_id = study.name 
        study_path = study.parent
        study_path_relative = study_path.relative_to(data_folder)
        logging.info("=========== PROCESSING SUBJECT: {} ===============".format(subject_id))
        out_dir = out_images_dir  / study_path_relative / subject_id
        preprocessus.checkDir(out_dir)
        
        csv_file = open(str(out_dir / 'info.csv'), 'w')
        field_names = ['File', 'type', 'tag']
        csvfilewriter = csv.DictWriter(csv_file, field_names)
        csvfilewriter.writeheader()
        
        i=1
        file_names = list( study.glob('**/*.dcm') )
        
        for file_name in file_names:

            logging.debug("============ FILE {} ==================== ".format(i))
            logging.debug(str(file_name))
            np_frame, us_type, capture_model = preprocessus.extractImageArrayFromUS(file_name, out_dir, None)
            
            # Extract text from the image
            tag = 'Unknown'
            if us_type not in non_tag_us and capture_model in tag_bounding_box.keys():
                tag = extracttagtext.extractTagFromFrame(np_frame, tag_bounding_box[capture_model], tag_list)
            tag_statistic[tag] += 1
            
            if capture_model not in tag_bounding_box.keys():
                logging.warning('========= US Model: {} was not found ========='.format(capture_model))

            csvfilewriter.writerow({'File': str(file_name), 'type': us_type, 'tag': tag})
            i+=1
        csv_file.close()

    pprint(tag_statistic)
    logging.info('---- DONE ----')


if __name__=="__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with subject subfolders.'
                'Every lowest level subfolder will be considered as a study')
    parser.add_argument('--out_dir', type=str, help='Output directory location.'
                'The directory hierarchy will be copied to this directory')
    args = parser.parse_args()

    main(args)