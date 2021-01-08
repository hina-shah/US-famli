from pathlib import Path
from argparse import ArgumentParser
import logging
import os
import time
import csv
import shutil
import sys
import SimpleITK as sitk
import utils
from FamliFileIO import taginfo
from FamliFileIO import csvwrap

def getTagDirPathListFile(out_folder, tag):
    """
    Sometimes OCR is unable to recognize some characters, we use permutations to identify them.
    The replacements here will put those permutations back in place.
    Ideally this should happen in a more generalized way.s
    """
    replacements = { "LI": "L1",
                    "RI": "R1",
                    "RI5": "R15",
                    "LI5": "L15",
                    "CI": "C1",
                    "LO": "L0",
                    "RO": "R0",
                    "3DI": "3D1"
                    }
    if tag in replacements:
        tag = replacements[tag]
    out_folder_path = out_folder/tag
    out_tag_list_file_path = out_folder_path / (tag + '_full_paths.txt')
    return out_folder_path, out_tag_list_file_path, tag

def main(args):
    data_folder = Path(args.dir)
    out_folder = Path(args.out_dir)
    utils.checkDir(out_folder, False)

    #  Setup logging:
    utils.setupLogFile(out_folder, args.debug)
    if args.cine_mode:
        tags = utils.getCineTagsList(args.tags)
    else:
        tags = utils.getTagsList(args.tags)
    print('Tags: {}'.format(tags))

    try:
        for tag in tags:
            out_folder_tag, out_tag_list_file_path, out_tag = getTagDirPathListFile(out_folder, tag)
            utils.checkDir(out_folder_tag, args.delete_existing)

    except Exception as e:
        logging.error("Couldn't split the tags string: {}".format(e))
        return

    gt_ga = {}
    if args.gt_ga_list:
        try:
            with open(args.gt_ga_list) as f:
                csv_reader = csv.DictReader(f)
                for line in csv_reader:
                    gt_ga[line['StudyID']] = {}
                    if line['ga_boe'] != ".":
                        gt_ga[line['StudyID']]['ga_boe'] = int(line['ga_boe'])
                    else:
                        gt_ga[line['StudyID']]['ga_boe'] = -1
                    
                    if line['ga_avua'] != ".":
                        gt_ga[line['StudyID']]['ga_avua'] = int(line['ga_avua'])
                    else:
                        gt_ga[line['StudyID']]['ga_avua'] = -1
        except OSError as e:
            logging.error('Error reading the gt ga file {} \n Error: {}'.format(args.gt_ga_list, e))
            gt_ga = {}
        print('Found {} studies with GT Ga'.format(len(gt_ga)))

    bounding_box = [[0,0], [255,250]]
    # Find all the info.csv files:
    tag_file_names = list( data_folder.glob('**/' + taginfo.TagInfoFile.file_name ))
    tag_file_list_rows = []
 
    for tag_file in tag_file_names:
        logging.info('--- PROCESSING: {}'.format(tag_file))
        files_to_copy = []
        tag_file_info = taginfo.TagInfoFile(tag_file.parent)
        tag_file_info.read()
        file_tag_pairs = tag_file_info.getFileNamesWithTags(tags)
        
        if len(file_tag_pairs) == 0:
            continue
        # print(file_tag_pairs[0])

        for file_tag_dict in file_tag_pairs:
            file_name = Path(file_tag_dict['File']).name
            name_no_suffix = Path(file_name).stem
            jpg_file_name = tag_file.parent/(name_no_suffix+'.jpg')

            cropped = None
            if jpg_file_name.exists():
                simage = sitk.ReadImage(str(jpg_file_name))
                if args.crop_images:        
                    size = simage.GetSize()
                    cropped = sitk.Crop(simage, bounding_box[0],
                                [size[i] - bounding_box[1][i] for i in range(2)])
                else:
                    cropped = simage
            
            tag = file_tag_dict['tag']
            tag_folder, out_tag_list_file_path, out_tag = getTagDirPathListFile(out_folder, tag)
            
            target_simlink_name = tag_folder/file_name

            # Get the data for the global list
            if args.create_global_list:
                if target_simlink_name.exists():
                    tag_file_row = {}
                    study_name = (tag_file.parent).name
                    pos = study_name.find('_')
                    if pos == -1:
                        logging.warning("Study name in path {} not in the correct format for a valid study".format(study_path))
                        continue

                    study_id = study_name[:pos]
                    study_date = study_name[pos+1:pos+9]
                    tag_file_row['study_id'] = study_id
                    tag_file_row['study_date'] = study_date
                    if len(gt_ga) > 0 and study_id in gt_ga:
                        tag_file_row['ga_boe'] = str(gt_ga[study_id]['ga_boe']) if gt_ga[study_id]['ga_boe'] > 0 else ''
                        tag_file_row['ga_avua'] = str(gt_ga[study_id]['ga_avua']) if gt_ga[study_id]['ga_avua'] > 0 else ''
                    else:
                        tag_file_row['ga_boe'] = ''
                        tag_file_row['ga_avua'] = ''

                    tag_file_row['file_path'] = target_simlink_name
                    tag_file_row['tag'] = out_tag
                    tag_file_list_rows.append(tag_file_row)
                else:
                    logging.info('The file: {}, study id: {} does not exist'.format(target_simlink_name, (tag_file.parent).name))
                continue

            # If not in global list generation mode, deal with the file based on what has been requested.
            out_jpg_name = tag_folder/(name_no_suffix+'.jpg')
            if os.path.exists(target_simlink_name):
                # count all files with that link
                logging.info('<---Found duplicates! ----> ')
                ext = Path(file_name).suffix
                all_target_simlink_files = list( Path(tag_folder).glob(stem+'*'+ext) )
                new_name = stem+'_'+str(len(all_target_simlink_files))+ext
                target_simlink_name = tag_folder/new_name
                new_name = stem+'_'+str(len(all_target_simlink_files))+'.jpg'
                out_jpg_name = tag_folder/(new_name+'.jpg')
            
            if cropped is not None:
                logging.info('Writing jpg image: {}'.format(out_jpg_name))
                sitk.WriteImage(cropped, str(out_jpg_name))
            
            source_file = Path(args.som_home_dir) / Path(file_tag_dict['File'])
            if not args.create_only_lists:
                logging.info('Copying file: {} -> {}, study:{}'.format(file_name, target_simlink_name, (tag_file.parent).stem))
                try:
                    shutil.copyfile(source_file, target_simlink_name)
                except FileNotFoundError:
                    logging.warning("Couldn't find file: {}".format(file))
                    continue
                except PermissionError:
                    logging.warning("Didn't have enough permissions to copy to target: {}".format(target_simlink_name))
                    continue
            else:
                with open(out_tag_list_file_path, "a") as fh:
                    fh.write( str(source_file) + "\n" )

    if args.create_global_list and len(tag_file_list_rows) > 0:
        logging.info('Number of tag file rows: {}, writing'.format( len(tag_file_list_rows)))
        outfilepath =  out_folder/'all_files_gt_ga.csv'
        try:
            csvwrap.CSVWrap.writeCSV(tag_file_list_rows, outfilepath)
        except IOError as e:
            logging.error('Error writing the output file: {} \n Error: {}'.format( outfilepath, e))
    logging.info('----- DONE -----')

if __name__=="__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with subject subfolders containing info.csv tagfiles.'
                'Every lowest level subfolder will be considered as a study', required=True)
    parser.add_argument('--out_dir', type=str, help='Output directory location.', required=True)
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    parser.add_argument('--tags', type=str, help='Space delimited list of tag files to be copied')
    parser.add_argument('--crop_images', action='store_true', help='Crop images while copying them over')
    parser.add_argument('--cine_mode', action='store_true', help='If option given, will copy only cines')
    parser.add_argument('--som_home_dir', type=str, help='SOM server directory, needed to replace a mount position in the file names', required=True)
    parser.add_argument('--create_only_lists', action='store_true', help='If option given, will only create extracted regions, and a list with full paths to dicoms.'
                                                                    ' These can be used to copy the files to another location.')
    parser.add_argument('--delete_existing', action='store_true', help='If option given, will delete existing directories')
    
    parser.add_argument('--create_global_list', action='store_true', help='Global list creation mode, this will create a global list of files, and assign a ga to each file.')
    parser.add_argument('--gt_ga_list', type=str, help='Path to the csv with ground truth GA by Study IDs')
    
    args = parser.parse_args()

    main(args)