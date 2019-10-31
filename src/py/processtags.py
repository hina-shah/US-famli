from pathlib import Path
from argparse import ArgumentParser
import logging
import os
import time
import csv
import shutil
import sys
import SimpleITK as sitk

def checkDir(dir_path, delete=True):
    # iF the output directory exists, delete it if requested
    if delete is True and dir_path.is_dir():
        shutil.rmtree(dir_path)
    
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)


def main(args):
    data_folder = Path(args.dir)
    out_folder = Path(args.out_dir)
    checkDir(out_folder)

    #  Setup logging:
    loglevel = logging.DEBUG if args.debug else logging.INFO
    log_file_name = "log" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
    logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S',
                         level=loglevel, filename=out_folder/log_file_name)
    tags=[]
    try:
        tags = args.tags.split(' ')
        logging.info('Looking for tags: {}'.format(tags))
        for tag in tags:
            checkDir(out_folder/tag, False)
    except:
        logging.error("Couldn't split the tags string")
        return

    bounding_box = [[0,0], [255,190]]
    # Find all the info.csv files:
    tag_file_names = list( data_folder.glob('**/info.csv') )
    for tag_file in tag_file_names:
        logging.info('--- PROCESSING: {}'.format(tag_file))
        files_to_copy = []
        try:
            with open(tag_file) as f:
                csv_reader = csv.DictReader(f)
                file_tag_pairs = [ (line['File'], line['tag']) for line in csv_reader if line['tag'] in tags ]
        except (OSError) as e:
            logging.error('Error reading csv file: {}'.format(tag_file))
            return

        for file, tag in file_tag_pairs:
            file_name = Path(file).name
            stem = Path(file_name).stem

            jpg_file_name = tag_file.parent/(stem+'.jpg')
            cropped = None
            if jpg_file_name.exists():
                simage = sitk.ReadImage(str(jpg_file_name))
                size = simage.GetSize()
                cropped = sitk.Crop(simage, bounding_box[0],
                            [size[i] - bounding_box[1][i] for i in range(2)])

            tag_folder = out_folder/tag
            target_simlink_name = tag_folder/file_name
            out_jpg_name = tag_folder/(stem+'.jpg')
            if os.path.exists(target_simlink_name):
                # count all files with that link
                logging.info('<---Found duplicates! ----> ')
                ext = Path(file_name).suffix
                all_target_simlink_files = list( Path(tag_folder).glob(stem+'*'+ext) )
                new_name = stem+'_'+str(len(all_target_simlink_files))+ext
                target_simlink_name = tag_folder/new_name
                new_name = stem+'_'+str(len(all_target_simlink_files))+'.jpg'
                out_jpg_name = tag_folder/(new_name+'.jpg')
            logging.info('Copying file: {} -> {}'.format(file, target_simlink_name))
            
            try:
                shutil.copyfile(file, target_simlink_name)
            except FileNotFoundError:
                logging.warning("Couldn't find file: {}".format(file))
            except PermissionError:
                logging.warning("Didn't have enough permissions to copy to target: {}".format(target_simlink_name))

            if cropped is not None:
                logging.info('Writing jpg image: {}'.format(out_jpg_name))
                sitk.WriteImage(cropped, str(out_jpg_name))
    logging.info('----- DONE -----')

if __name__=="__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with subject subfolders containing info.csv tagfiles.'
                'Every lowest level subfolder will be considered as a study')
    parser.add_argument('--out_dir', type=str, help='Output directory location.')
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    parser.add_argument('--tags', type=str, help='Space delimited list of tag files to be copied')
    args = parser.parse_args()

    main(args)