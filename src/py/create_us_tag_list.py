#!/usr/bin/env python

'''
This is a script to run OCR on all the study folders in the input directory.
It replicates the directory structure in the output directory.
- The output directory structure also has info.csv in each study folder which lists the image type, and tag for the file
- The tags are read from a specified bounding box in the ultrasound image
- For cines, the tag is extracted from the middle frame of the video
- If the directory structure already exists, this won't overwrite the results
- Parallel processing is used whenever possible to speed up the process.

Author: Hina Shah
'''

from pathlib import Path
import os
import csv
from argparse import ArgumentParser
from pprint import pprint, pformat
import logging
import sys
import gc
import concurrent.futures
import time
import json
import numpy as np

from FamliImgIO import dcmio
from FamliFileIO import taginfo
from FamliOCR import famlitesseract as tess
import utils

def getStudyOutputFolder(study, data_folder, out_parent_folder):
    subject_id = study.name 
    study_path = study.parent
    study_path_relative = study_path.relative_to(data_folder)
    out_dir = out_parent_folder  / study_path_relative / subject_id
    return out_dir

def extractImageArrayFromUS(file_path, out_dir=None):
    dcmobj = dcmio.DCMIO(file_path)
    # Create the output path
    out_path = None
    if out_dir is not None:
        out_path = out_dir / (file_path.stem + '.jpg')
    
    np_frame = dcmobj.get_repres_frame(out_path)
    us_type = dcmobj.get_type()
    us_model = dcmobj.get_model()

    return np_frame, us_type, us_model

def extractTagForStudy(study, data_folder, out_images_dir, tag_list, non_tag_us, 
                        tag_bounding_box, server_path, greedy=False):
    
    logging.info("=========== PROCESSING SUBJECT: {} ===============".format(study.name))
    out_dir = getStudyOutputFolder(study, data_folder, out_images_dir)
    utils.checkDir(out_dir, delete = (not greedy))
    
    # look for the info.csv file in the output folder. 
    # if exists, read the file names and the tags
     
    i=1
    file_paths = list( study.glob('**/*.dcm') )
    csvrows=[]
    unknown = 0
    tag_manager = taginfo.TagInfoFile(out_dir)
    tag_statistic = {}
    prev_tags = {}
    if greedy and tag_manager.exists():
        # If this is in greedy mode, try to get the names of the files
        # that have an undecided tag:
        tag_manager.read()
        # get a copy of the previously created tags for all files
        prev_tags = tag_manager.getDictWithNameKeys()
        # clear the tags stored in the file. (This does not delete the file, just empties the data structure)
        tag_manager.clear()
        tag_manager.deleteTagFile()

    for file_path in file_paths:
        logging.debug("FILE {}: {}".format(i, str(file_path)))

        # Make sure path exists
        if not file_path.exists():
            logging.warning('File: {} does not exist, skipping'.format(file_path))
            continue

        # if in greedy mode, look at the tag, look at the tag and decide if we wamt to extract tag
        getTag = True
        tag = None
        us_type = None
        if greedy and len(prev_tags) > 0:
            name = file_path.name
            if name in prev_tags:
                tag = prev_tags[name][1]
                us_type = prev_tags[name][0]
                if tag in ['Unknown', 'Undecided', 'No tag']:
                    getTag = True
                else:
                     getTag = False

        if getTag:
            start = time.time()
            # Get the representative frame
            np_frame, us_type, capture_model = extractImageArrayFromUS(file_path, out_dir=out_dir)
            end = time.time()
            logging.debug('Preprocessing took : {} seconds'.format(end-start))
            if len(capture_model)>0 and capture_model not in tag_bounding_box.keys():
                logging.warning('US Model: {} not supported for file: {}'.format(capture_model, file_path))
                del np_frame
                continue
            # Extract the tag
            start = time.time()
            tag = 'Unknown'
            if np_frame is not None and \
                us_type not in non_tag_us and \
                capture_model in tag_bounding_box.keys():
                # Run tag extraction
                tag = tess.extractTagFromFrame(np_frame, tag_bounding_box[capture_model], tag_list)
            end = time.time()
            logging.debug('Tag extraction took : {} seconds'.format(end-start))
            del np_frame
            if tag in ['Unknown', 'Undecided', 'No tag']:
                unknown += 1
        else:
            logging.debug('Skipping the file: {}, tag: {}, type: {}, because it was known'.format(file_path, tag, us_type))

        tag_manager.addTag(file_path.parent, server_path, file_path.name, us_type, tag, write=True)
        i+=1
        gc.collect()

    tag_statistic = tag_manager.tag_statistic
    # If all unknown, delete the tag file. 
    if unknown == len(file_paths):
        tag_manager.deleteTagFile()
        tag_manager.clear()
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
    
    # read the list of acceptable tags in the ultrasound file
    tag_list = utils.getTagsList()
    tag_statistic = dict.fromkeys(tag_list, 0)
    tag_statistic['Unknown'] = 0
    tag_statistic['Undecided'] = 0
    tag_statistic['No tag'] = 0
    
    # Approximate bounding box of where the tag is written acoording to the 
    # us model
    tag_bounding_box = { 'V830':[[40,75], [255,190]],
                         'LOGIQe':  [[0,55], [200,160]],
                         'Voluson S': [[40,75], [255,190]],
                         'LOGIQeCine': [[0,0],[135,90]],
                         'Turbo': [[75,20,],[230,80]],
                         'Voluson E8': [[40,75], [255,250]]
                        }

    # list of ultrasound image types whose tags we do not care about right now.
    non_tag_us = ['Unknown', 'Secondary capture image report',
                    'Comprehensive SR', '3D Dicom Volume']

    
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
                tag_file_man =taginfo.TagInfoFile(infocsv_dir)
                tag_file_man.read()
                if tag_file_man.getNumFiles() > 0:
                    for tag in tag_file_man.tag_statistic:
                        if tag not in tag_statistic:
                            tag_statistic[tag] = 0
                        tag_statistic[tag] += tag_file_man.tag_statistic[tag]        
            except (OSError, ValueError) as err:
                logging.warning('Error reading previously created tags.csv for subject: {}: {}'.format(study, err))
            except:
                logging.warning('Error reading previously created tags.csv for subject: {}'.format(study))
                logging.warning('Unknown except while reading csvt: {}'.format(sys.exc_info()[0]))
    else:
        cleaned_studies = studies
    del studies

    if args.use_threads:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Start the load operations and mark each future with its URL
            future_tags = {executor.submit(extractTagForStudy, study, 
                                            data_folder, out_images_dir, tag_list,
                                            non_tag_us, tag_bounding_box, 
                                            Path(args.server_path), args.greedy ): study for study in cleaned_studies}
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
                                                    tag_list, non_tag_us, tag_bounding_box, 
                                                    Path(args.server_path), args.greedy)
            logging.info('Finished processing: {}'.format(study))
            for key, value in this_tag_statistic.items():
                tag_statistic[key] += value
            endstr = "\n" if i%50 == 0 else "."
            print("",end=endstr)
            with open(finished_study_file, "a+") as f:
                f.write(str(study)+os.linesep)
            i+=1
    
    pprint(tag_statistic)
    with open(out_images_dir/"NumberOfTags.json", "w") as outfile:
        json.dump(tag_statistic, outfile, indent=4) 
    logging.info(pformat(tag_statistic))
    logging.info('---- DONE ----')
    print('------DONE-----------')


if __name__=="__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with subject subfolders.'
                'Every lowest level subfolder will be considered as a study', required=True)
    parser.add_argument('--out_dir', type=str, help='Output directory location.'
                'The directory hierarchy will be copied to this directory', required=True)
    parser.add_argument('--server_path', type=str, help='Path to the server relative to which teh paths of files will be saved')
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    parser.add_argument('--use_threads', action='store_true', help='Use threads to run the code')
    parser.add_argument('--greedy', action='store_true', help='If given, will retry to extract tags for files that were processed previously')
    args = parser.parse_args()

    main(args)