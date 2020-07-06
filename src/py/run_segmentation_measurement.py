from pathlib import Path
import os
import csv
from argparse import ArgumentParser
import logging
import sys
import multiprocessing as mp
import time
import utils
import json
import subprocess
import shutil

def processDir(dir_path):
    # Does directory exist
    path_exists = dir_path.exists()
    if path_exists:
        # if it exists how many files exist here?
        path_exists = len(os.listdir(str(dir_path))) > 0
    print('Process: {} ? {}'.format(dir_path, not path_exists))
    return not path_exists

def runSegmentationOnFolder(data_folder, path_to_classification='/HOMER_STOR/hinashah', path_to_fit_sh = '/HOMER_STOR/hinashah/predict-sh'):
    output_folder_path = Path(data_folder+'_seg')
    output_fit_folder_path = Path(data_folder+'_seg_fit')
    class_name = Path(data_folder).stem
    class_type_map = {}
    class_type_map['AC'] = ['US-famli_predict_ac-mask', 'ac-mask-predict.sh']
    class_type_map['HC'] = ['US-famli_predict_bpd-mask', 'bpd-mask-predict.sh']
    class_type_map['FL'] = ['US-famli_predict_fl-mask', 'fl-mask-predict.sh']

    if class_name in class_type_map:
        
        if processDir(output_folder_path):
            logging.info('Running Segmentation for: {}'.format(data_folder))
            node_source_path = path_to_classification + '/us-famli-nn/bin/index.js'
            utils.checkDir(output_folder_path, False)
            command_list = ['node', node_source_path, '--dir', str(data_folder), '--type', 'remove_calipers', '--type', class_type_map[class_name][0], '--out', str(output_folder_path)]
            print(command_list)
            try:
                subprocess.run(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
            except Exception as e:
                logging.error(e)
                logging.info('Error processing image dump folder: {}'.format(data_folder))
                return 0
        
        if processDir(output_fit_folder_path):
            logging.info('Running Fitting for: {}'.format(data_folder))
            utils.checkDir(output_fit_folder_path, False)
            cmd_path = str(Path(path_to_fit_sh) / class_type_map[class_name][1])
            command_list = [cmd_path, '-i', str(output_folder_path), '-o', str(output_fit_folder_path)]
            try:
                subprocess.run(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
            except Exception as e:
                logging.error(e)
                logging.info('Error processing Segmentation folder: {}'.format(output_folder_path))
                return 0

def main(args):
    data_folder = Path(args.dir)
    out_images_dir = Path(args.out_dir)

    utils.checkDir(out_images_dir, False)
    utils.setupLogFile(out_images_dir, args.debug)

    predictions_csv_list = []
    out_suffix = ''
    if args.include_dirs_list is not None:
        logging.info('---- PROCESSING BATCH FILE: {} -------'.format(args.include_dirs_list))
        out_suffix = '_' + Path(args.include_dirs_list).stem
        include_dirs = []
        try:
            with open(args.include_dirs_list) as f:
                lines = f.readlines()
                include_dirs = [x.strip() for x in lines]
        except Exception as e:
            logging.error('Error while opening the include dir list: {}'.format(e))
            return
       
        for sub_dir in include_dirs:
            sub_dir = sub_dir.strip('\"')
            sub_dir_list = list( (data_folder/sub_dir).glob('**/prediction_*.csv') )
            predictions_csv_list.extend(sub_dir_list)
    else:
        predictions_csv_list = list( data_folder.glob('**/prediction_*.csv') )

    logging.info('Found {} prediction files '.format(len(predictions_csv_list)))
    print('Found {} prediction files'.format(len(predictions_csv_list)))  

    class_biom_map = {}
    class_biom_map['head'] = 'HC'
    class_biom_map['femur'] = 'FL'
    class_biom_map['abdomen'] = 'AC'
    class_biom_map['fetus'] = 'CRL'

    class_db = {}
    threshold = 0.9
    for i, pred_file in enumerate(predictions_csv_list):
        print('Processing file {}/{}'.format(i, len(predictions_csv_list)))
        logging.info('--- PROCESSING: {}'.format(pred_file))
        
        if not pred_file.exists() or os.stat(str(pred_file)).st_size < 100:
            logging.warning('Prediction file seems to be incomplete, skipping: {}'.format(pred_file))
            print('Skipping {}, size: {}'.format(pred_file, os.stat(str(pred_file)).st_size if pred_file.exists() else -1))
            continue

        study_name = pred_file.parent.stem
        study_path = pred_file.parent
        if study_name not in class_db:
            class_db[study_name] = {}
            class_db[study_name]['class_path'] = str(study_path)
             # create a folder for that class in the output folder
            for class_name in class_biom_map:
                mapped_name = class_biom_map[class_name]
                utils.checkDir(study_path/mapped_name , False)
                class_db[study_name][mapped_name] = {}

                class_db[study_name][mapped_name]['path'] = str(study_path/mapped_name)
                class_db[study_name][mapped_name]['images'] = []
        try:
            with open(pred_file) as f:
                # Read the prediction file
                pred_file_reader = csv.DictReader(f)
                # For each frame, find the classification class, and 
                # pick it if the probability is more than 0.9
                for line in pred_file_reader:
                    class_name = line['class']
                    if float(line[class_name]) > threshold:
                        # create a symbolic link for the image
                        if line['img'].find('dataset_C1_HC_fullset') > -1:
                            line['img'] = line['img'].replace('dataset_C1_HC_fullset', data_folder.stem)

                        frame_path = Path(line['img'])
                        target_frame_name = frame_path.parent.stem + '_' + frame_path.name
                        target_frame_path = study_path/class_biom_map[class_name]/target_frame_name
                        if not target_frame_path.exists():
                            if target_frame_path.is_symlink():
                                print('Removing the link: {}'.format(target_frame_path))
                                os.unlink(str(target_frame_path))
                            os.symlink(line['img'], target_frame_path)
                        # store the path to the folder and symbolic link in class_db
                        class_db[study_name][class_biom_map[class_name]]['images'].append(str(target_frame_path))
        except Exception as e:
            logging.error('Error in processing prediction file: {} \n Error: {}'.format(pred_file, e))
    
    try:
        j_f = json.dumps(class_db, indent=4)
        f = open(str(out_images_dir/ ('classification_details' + out_suffix + '.json')),"w")
        f.write(j_f)
        f.close()
    except Exception as e:
        print(e)
        logging.error('Error dumping the json file')
        return
    
    # Run segmentation and size estimation on each study
    print('****** ONTO SEGMENTATION AND FITTING *****')
    for i, study in enumerate(class_db):
        print('Segmentation/Fitting Processing study {}/{}, Name: {}'.format(i, len(class_db.keys()), study))
        logging.info('--- Segmentation/Fitting PROCESSING: {}'.format(study))
        class_keys = [key for key in class_db[study] if 'images' in class_db[study][key]]
        
        # for each of these run the segmentation, fitting code in parallel
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(runSegmentationOnFolder, [class_db[study][class_name]['path'] for class_name in class_keys])
        pool.close()
    print('------DONE-----------')

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with study folders that have prediction.csv files generated', required=True)
    parser.add_argument('--out_dir', type=str, help='Output directory location.'
                                'The directory hierarchy will be copied to this directory', required=True)
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    parser.add_argument('--include_dirs_list', default=None, help='A text file with the directories to process in this run')
    args = parser.parse_args()

    main(args)