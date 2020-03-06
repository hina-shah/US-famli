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
from dump_images_from_cine import dumpImagesForCine
import subprocess
import shutil
import SimpleITK as sitk

def runClassificationOnFolder(data_folder, path_to_classification, out_folder, 
                                class_name, prediction_threshold=0.9, 
                                    sitk_reference_image_path=None):
    """
    Run a node-based classification script on the input folder.
    The classification (javascript) script's source code is at: https://github.com/juanprietob/us-famli-nn
    Input parameters:
    data_folder : Points to the directory that has the image dump for all the frames of a cine (basically a folder of images)
    path_to_classification: Points to the directory where us-famli-nn has been set up
    out_folder: Points to the output directory. a prediction_<data_folder_name>.csv file will be written out here.
    class_name: Name of the class to look for in the generated predictions file
    prediction_threshold: probability threshold to keep/copy the images
    """
    node_source_path = path_to_classification + '/us-famli-nn/bin/index.js'
    if not Path(node_source_path).exists():
        logging.error('Source code for classification does not exist at: {}'.format(node_source_path))
        return 0
    
    output_file_path = out_folder / ('prediction_' + data_folder.stem + '.csv') 
    output_file = str(output_file_path)
    if output_file_path.exists():
        logging.debug('Prediction file: {} exists, skipping the corresponding folder'.format(output_file_path))
        return 0

    command_list = ['node', node_source_path, '--dir', str(data_folder), '--type', 'remove_calipers', '--type', 'classify_ga', '--out', output_file]
    try:
        subprocess.run(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    except Exception as e:
        logging.error(e)
        logging.info('Error processing image dump folder: {}'.format(data_folder))
        return 0
    
    class_images = []
    try:
        with open(output_file) as f:
            # Read prediction.csv file
            csv_reader = csv.DictReader(f)
            # Keep files with prediction > threshold and class class_nam
            class_images = [line['img'] for line in csv_reader if  \
                            line['class'] == class_name and \
                            float(line[class_name]) > prediction_threshold]
    except Exception as e:
        logging.error(e)
        logging.info('Error processing the prediction file {}'.format(output_file))
        return 0

    if len(class_images) > 0:
        copy_dest_dir = out_folder / 'cine_class_images'
        utils.checkDir(copy_dest_dir, False)
        if sitk_reference_image_path is not None:
            ref_img = sitk.ReadImage(str(sitk_reference_image_path))
            ref_spacing = ref_img.GetSpacing()[:2]
            ref_direction = ref_img.GetDirection()
            target_direction = [ref_direction[0], ref_direction[1], ref_direction[3], ref_direction[4]]

        for image_path_str in class_images:
            image_path = Path(image_path_str)
            dest_file_path = str(copy_dest_dir / (data_folder.stem + '_' + image_path.name))
            if sitk_reference_image_path is not None:            
                src_image = sitk.ReadImage(image_path_str)
                if src_image.GetDimension() == 3:
                    logging.warning('GOT a 2d image with dimension 3 {}'.format(src_image.GetSize()))

                target_size = [  sz*ss/rs for sz, ss, rs in zip( src_image.GetSize()[:2], src_image.GetSpacing()[:2], ref_img.GetSize()[:2]) ]
                print(target_size)

                sitk_resample = sitk.ResampleImageFilter()
                sitk_resample.SetOutputSpacing(ref_spacing)
                sitk_resample.SetOutputDirection(target_direction)
                sitk_resample.SetSize(target_size)
                sitk_out_img = sitk_resample.Execute(src_image)
                sitk.WriteImage(sitk_out_img, dest_file_path)
                del src_image
                del sitk_out_img
                del sitk_resample
            else:
                shutil.copyfile(image_path_str, dest_file_path)
    return len(class_images)

def getStudyOutputFolder(study, data_folder, out_parent_folder):
    """
    Creates a path for a study that is relative to the output parent directory.
    This is useful when copying the tree structure of the studies

    For example:
    study = /a/b/c/Ultrasounds/2019-10/UNC-001
    data_folder = /a/b/c
    out_parent_folder = /d/e/f/g

    this returns: /d/e/f/g/Ultrasounds/2019-10/UNC-001
    """
    subject_id = study.name
    study_path = study.parent
    study_path_relative = study_path.relative_to(data_folder)
    out_dir = out_parent_folder  / study_path_relative / subject_id
    return out_dir

def isStudyCompletelyProcessed(out_study_dir, bs_list):
    study_processed = True
    study_processed = study_processed & (out_study_dir/'head_2d_image.dcm').exists()
    for bs_file in bs_list:
        cine_name = Path(bs_file['File']).stem
        prediction_file = out_study_dir/ ('prediction_' + cine_name + '.csv')
        study_processed = study_processed & prediction_file.exists()

    return study_processed

def main(args):
    data_folder = Path(args.dir)
    out_images_dir = Path(args.out_dir)

    utils.checkDir(out_images_dir, False)
    utils.setupLogFile(out_images_dir, args.debug)

    # read the list of acceptable tags in the ultrasound file
    tags = ['M', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'R15', 'R45', 'R0', 'RO', 'R1',
                    'L15', 'L45', 'L0', 'LO', 'L1', 'M0', 'M1', 'RTA', 'RTB', 'RTC']
    tag_2d = ['HC']

    info_csv_list = []
    out_suffix = ''
    if args.include_dirs_list is not None:
        out_suffix = '_' + Path(args.include_dirs_list).stem
        include_dirs = []
        try:
            with open(args.include_dirs_list) as f:
                lines = f.readlines()
                include_dirs = [x.strip() for x in lines]
        except Exception as e:
            logging.error('Error while opening the include dir list: {}'.format(e))
            return
        all_csvs = list( data_folder.glob('**/info_corrected.csv') )
        for sub_dir in include_dirs:
            sub_dir_list = [csv_file for csv_file in all_csvs if csv_file.match('**/' + sub_dir + '/*/*.csv')]
            info_csv_list.extend(sub_dir_list)
    else:
        info_csv_list = list( data_folder.glob('**/info_corrected.csv') )

    logging.info('Found {} studies with an info_corrected.csv '.format(len(info_csv_list)))
    print('Found {} studies with an info_corrected.csv '.format(len(info_csv_list)))            

    img_count_db = {}
    bs_db = {}
    counter = 0
    for i, tag_file in enumerate(info_csv_list):
        print('Processing file {}/{}, found {} studies with BS and HC'.format(i, len(info_csv_list), counter))
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
                logging.debug('BS LIST')
                logging.debug(bs_list)
                logging.debug('IM LIST')
                logging.debug(im_list)
                if len(bs_list) > 0 and len(im_list) > 0:
                    study_name = (tag_file.parent).stem
                    study_detail = {}
                    study_detail['blindsweeps'] = bs_list
                    study_detail['hc_2d'] = im_list
                    bs_db[study_name] = study_detail
                    # Create the study directory in the output folder
                    out_study_dir = getStudyOutputFolder(tag_file.parent, data_folder, out_images_dir)
                    utils.checkDir(out_study_dir, False)
                    logging.debug('Output folder is: {}'.format(out_study_dir))

                    # Continue if the processing has already been done on this folder:
                    if isStudyCompletelyProcessed(out_study_dir, bs_list):
                        logging.info('Output study: {} already processed, SKIPPING'.format(out_study_dir))
                        continue

                    hc_img_path = Path(args.som_working_dir + '/' + im_list[0]['File'])
                    # Iterate through the blindsweeps list
                    num_imgs_found = 0
                    for bs_item in bs_list:
                        # Dump images of the blindsweep -> in the form of nrrd's (not jpgs)
                        img_dump_dir = out_study_dir/'ImgDump'
                        utils.checkDir(img_dump_dir, False)
                        logging.debug('Dumping images for cine: {}'.format(bs_item['File']))
                        cine_dump_path = dumpImagesForCine( Path( args.som_working_dir + '/' + bs_item['File']), img_dump_dir)
                        # Run classification on the imgdump folder
                        logging.debug('Run classification for cine: {}'.format(bs_item['File']))
                        num_imgs_found += runClassificationOnFolder(Path(cine_dump_path), args.path_to_classification, Path(out_study_dir), \
                                                                    class_name='head', sitk_reference_image_path=hc_img_path)
                    logging.debug('Number of head images found in blindsweeps: {}'.format(num_imgs_found))
                    img_count_db[str(out_study_dir)] = num_imgs_found
                    # Copy the first head images to a 2d_class_images
                    logging.debug('Copying the first HC file: {}'.format(hc_img_path))
                    shutil.copyfile(hc_img_path, out_study_dir/'head_2d_image.dcm')
                    #Delete the imgdump folder
                    shutil.rmtree(img_dump_dir)
                    counter+=1
        except (OSError) as e:
            logging.error('Error reading csv file: {}'.format(tag_file))
            return
        except Exception as e:
            logging.error('Error in processing the tag file: {}'.format(e))
    
    # Write the list of HC and blindsweep images
    try:
        j_f = json.dumps(bs_db, indent=4)
        f = open(str(out_images_dir/ ('hc_and_blindsweeps' + out_suffix + '.json')),"w")
        f.write(j_f)
        f.close()
    except Exception as e:
        print(e)
        logging.error('Error dumping the json file')
        return

    # Write out the number of images identified in each study:
    try:
        j_f = json.dumps(img_count_db, indent=4)
        with open(str(out_images_dir/ ('study_images_count' + out_suffix + '.json')),"w") as f:
            f.write(j_f)
    except Exception as e:
        print(e)
        logging.error('Error dumping the image count file')

    logging.info('Found {} studies with HC files'.format(counter))
    print('Found {} studies with HC files'.format(counter))
    print('------DONE-----------')

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with study folders that have an info.csv file generated', required=True)
    parser.add_argument('--out_dir', type=str, help='Output directory location.'
                                'The directory hierarchy will be copied to this directory', required=True)
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    parser.add_argument('--som_working_dir', type=str, help = 'Mount location for the SOM server.' 
                                'Basically parent location for the famli folder, and original ultrasound images are stored here', required=True)
    parser.add_argument('--path_to_classification', type=str, help='Path to the classification code : us-famli-nn/bin/index.js')
    parser.add_argument('--use_threads', action='store_true', help='Use threads to run the code')
    parser.add_argument('--include_dirs_list', default=None, help='A text file with the directories to process in this run')
    args = parser.parse_args()

    main(args)