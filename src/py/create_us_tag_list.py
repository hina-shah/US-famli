#!/usr/bin/env python

from pathlib import Path
import os
import csv
from argparse import ArgumentParser
from pprint import pprint, pformat
import logging
import sys
import gc
import concurrent.futures
import preprocessus
import extracttagtext
import time
import utils

def getStudyOutputFolder(study, data_folder, out_parent_folder):
    subject_id = study.name 
    study_path = study.parent
    study_path_relative = study_path.relative_to(data_folder)
    out_dir = out_parent_folder  / study_path_relative / subject_id
    return out_dir

def extractTagForStudy(study, data_folder, out_images_dir, tag_list, non_tag_us, tag_bounding_box):
    tag_statistic = dict.fromkeys(tag_list, 0)
    tag_statistic['Unknown'] = 0
    tag_statistic['Undecided'] = 0
    tag_statistic['No tag'] = 0
    
    logging.info("=========== PROCESSING SUBJECT: {} ===============".format(study.name))
    out_dir = getStudyOutputFolder(study, data_folder, out_images_dir)
    utils.checkDir(out_dir)
    
    i=1
    file_names = list( study.glob('**/*.dcm') )
    csvrows=[]
    unknown = 0
    for file_name in file_names:

        start = time.time()
        logging.debug("FILE {}: {}".format(i, str(file_name)))
        logging.debug(str(file_name))
        np_frame, us_type, capture_model = preprocessus.extractImageArrayFromUS(file_name, out_dir, None)
        end = time.time()
        logging.debug('Preprocessing took : {} seconds'.format(end-start))
        
        if len(capture_model)>0 and capture_model not in tag_bounding_box.keys():
            logging.warning('US Model: {} not supported for file: {}'.format(capture_model, file_name))
            del np_frame
            continue

        # Extract text from the image
        start = time.time()
        tag = 'Unknown'
        if np_frame is not None and \
            us_type not in non_tag_us and\
            capture_model in tag_bounding_box.keys():
            # Run tag extraction
            tag = extracttagtext.extractTagFromFrame(np_frame, tag_bounding_box[capture_model], tag_list)
        end = time.time()
        logging.debug('Tag extraction took : {} seconds'.format(end-start))
        tag_statistic[tag] += 1
        del np_frame
        
        if tag in ['Unknown', 'Undecided', 'No tag']:
            unknown += 1

        csvrows.append({'File': str(file_name), 'type': us_type, 'tag': tag})
        i+=1
        gc.collect()

    # Write the csv file
    if len(csvrows) > 0 and (unknown < len(csvrows)):
        csv_file = open(str(out_dir / 'info.csv'), 'w')
        field_names = ['File', 'type', 'tag']
        csvfilewriter = csv.DictWriter(csv_file, field_names)
        csvfilewriter.writeheader()
        csvfilewriter.writerows(csvrows) 
        csv_file.close()
    return tag_statistic


def main(args):
    data_folder = Path(args.dir)
    out_images_dir = Path(args.out_dir)

    utils.checkDir(out_images_dir, False)    
    utils.setupLogFile(out_images_dir, args.debug)
    
    studies = []
    for dirname, dirnames, __ in os.walk(str(data_folder)):
        if len(dirnames) == 0:
            studies.append(Path(dirname))
            
    logging.info('Found {} studies '.format(len(studies)))
    print('Found {} studies '.format(len(studies)))
    
    tag_list_file = 'us_tags.txt'
    
    # Approximate bounding box of where the tag is written acoording to the 
    # us model
    tag_bounding_box = { 'V830':[[40,75], [255,190]],
                        'LOGIQe':  [[0,55], [200,160]],
                         'Voluson S': [[40,75], [255,190]]}

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
        tags = ['M', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'R15', 'R45', 'R0', 'RO', 'R1',
                    'L15', 'L45', 'L0', 'LO', 'L1', 'M0', 'M1', 'RTA', 'RTB', 'RTC']

    tag_statistic = dict.fromkeys(tag_list, 0)
    tag_statistic['Unknown'] = 0
    tag_statistic['Undecided'] = 0
    tag_statistic['No tag'] = 0

    # Also read in study directories that might have been finished by a previous run - do not want to rerun them again
    finished_study_file = out_images_dir/'finished_studies.txt'
    finished_studies = None
    if finished_study_file.exists():
        with open(finished_study_file) as f:
            finished_studies = f.read().splitlines()
            finished_studies = [study for study in finished_studies if study.strip()]
    if finished_studies is not None:
        logging.info('Found {} finished studies'.format(len(finished_studies)))
        cleaned_studies = [study for study in studies if str(study) not in finished_studies]
        # Get statistics for the finished studies
        for study in finished_studies:
            logging.info('Will skip: {}'.format(study))
            try:
                infocsv_dir = getStudyOutputFolder(Path(study), data_folder, out_images_dir)
                logging.info('Opening: {}'.format(infocsv_dir))
                with open(infocsv_dir/'info.csv', 'r') as f:
                    csv_reader = csv.DictReader(f)
                    for line in csv_reader:
                        tag_statistic[line['tag']] += 1
            except (OSError, ValueError) as err:
                logging.warning('Error reading previously created info.csv for subject: {}: {}'.format(study, err))
            except:
                logging.warning('Error reading previously created info.csv for subject: {}'.format(study))
                logging.warning('Unknown except while reading csvt: {}'.format(sys.exc_info()[0]))
    else:
        cleaned_studies = studies
    del studies

    if args.use_threads:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Start the load operations and mark each future with its URL
            future_tags = {executor.submit(extractTagForStudy, study, 
                                            data_folder, out_images_dir, tag_list,
                                            non_tag_us, tag_bounding_box): study for study in cleaned_studies}
            for future in concurrent.futures.as_completed(future_tags):
                d = future_tags[future] 
                logging.info('Finished processing: {}'.format(d))
                this_tag_statistic = future.result()
                #logging.info(future.result())
                for key, value in this_tag_statistic.items():
                    tag_statistic[key] += value
                with open(finished_study_file, "a+") as f:
                    f.write(str(d)+os.linesep)
    else:
        i=1
        for study in cleaned_studies:
            this_tag_statistic = extractTagForStudy(study, data_folder, out_images_dir, 
                                                    tag_list, non_tag_us, tag_bounding_box)
            logging.info('Finished processing: {}'.format(study))
            for key, value in this_tag_statistic.items():
                tag_statistic[key] += value
            endstr = "\n" if i%50 == 0 else "."
            print("",end=endstr)
            with open(finished_study_file, "a+") as f:
                f.write(str(study)+os.linesep)
            i+=1
    
    pprint(tag_statistic)
    logging.info(pformat(tag_statistic))
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