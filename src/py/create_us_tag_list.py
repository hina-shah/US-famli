#!/usr/bin/env python

from pathlib import Path
import os
import csv
from argparse import ArgumentParser
from pprint import pprint
import logging
import sys
import gc
import concurrent.futures
import preprocessus
import extracttagtext

def extractTagForStudy(study, data_folder, out_images_dir, tag_list, non_tag_us, tag_bounding_box):
    tag_statistic = dict.fromkeys(tag_list, 0)
    tag_statistic['Unknown'] = 0
    tag_statistic['Undecided'] = 0
    tag_statistic['No tag'] = 0
    
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

        logging.debug("FILE {}: {}".format(i, str(file_name)))
        logging.debug(str(file_name))
        np_frame, us_type, capture_model = preprocessus.extractImageArrayFromUS(file_name, out_dir, None)
        
        # Extract text from the image
        tag = 'Unknown'
        if us_type not in non_tag_us and capture_model in tag_bounding_box.keys():
            tag = extracttagtext.extractTagFromFrame(np_frame, tag_bounding_box[capture_model], tag_list)
        tag_statistic[tag] += 1
        del np_frame

        if len(capture_model)>0 and capture_model not in tag_bounding_box.keys():
            logging.warning('US Model: {} was not found for file: {}'.format(capture_model, file_name))

        csvfilewriter.writerow({'File': str(file_name), 'type': us_type, 'tag': tag})
        i+=1
        gc.collect()

    csv_file.close()
    return tag_statistic


def main(args):
    data_folder = Path(args.dir)
    out_images_dir = Path(args.out_dir)

    preprocessus.checkDir(out_images_dir, False)    
    loglevel = logging.DEBUG if args.debug else logging.INFO

    logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S',
                         level=loglevel, filename=out_images_dir/'log.txt')

    studies = []
    for dirname, dirnames, __ in os.walk(str(data_folder)):
        if len(dirnames) == 0:
            studies.append(Path(dirname))
            
    logging.info('Found {} studies '.format(len(studies)))
    print('Found {} studies '.format(len(studies)))
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

    if args.use_threads:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Start the load operations and mark each future with its URL
            future_tags = {executor.submit(extractTagForStudy, study, 
                                            data_folder, out_images_dir, tag_list,
                                            non_tag_us, tag_bounding_box): study for study in studies}
            for future in concurrent.futures.as_completed(future_tags):
                d = future_tags[future] 
                logging.info('Finished processing: {}'.format(d))
                this_tag_statistic = future.result()
                logging.info(future.result())
                for key, value in this_tag_statistic.items():
                    tag_statistic[key] += value
    else:
        for study in studies:
            this_tag_statistic = extractTagForStudy(study, data_folder, out_images_dir, tag_list, non_tag_us, tag_bounding_box)
            logging.info('Finished processing: {}'.format(study))
            for key, value in this_tag_statistic.items():
                tag_statistic[key] += value
    
    pprint(tag_statistic)
    logging.info('---- DONE ----')
    print('------DONE-----------')


if __name__=="__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with subject subfolders.'
                'Every lowest level subfolder will be considered as a study')
    parser.add_argument('--out_dir', type=str, help='Output directory location.'
                'The directory hierarchy will be copied to this directory')
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    parser.add_argument('--use_threads', action='store_true', help='Use threads to run the code')
    args = parser.parse_args()

    main(args)