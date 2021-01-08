from pathlib import Path
import os
import csv
from argparse import ArgumentParser
import logging
import utils
from FamliFileIO import taginfo, csvwrap

def main(args):
    tag_list = utils.getCineTagsList()

    data_folder = Path(args.dir)
    info_csv_list = []
    info_csv_list = list( data_folder.glob('**/' + taginfo.TagInfoFile.file_name) )

    study_rows = []
    for i,tag_file in enumerate(info_csv_list):
        print('Processing file {}/{}'.format(i, len(info_csv_list)))
        logging.info('--- PROCESSING: {}'.format(tag_file))

        tag_info = taginfo.TagInfoFile(tag_file.parent)
        study_name = tag_info.getStudyName()

        this_study_row={}
        parts = study_name.split('_')
        # Get the study id
        this_study_row['full_study_id'] = study_name # eg: UNC-0026-1_20180924_093918
        this_study_row['study_id'] = parts[0] # eg: UNC-0026-1
        pid = parts[0]
        this_study_row['pid'] = pid[:pid.rfind('-')] # eg: UNC-0026
        # Get the date of the study
        this_study_row['study_date'] = parts[1][4:6] + '/' + parts[1][6:8] + '/' + parts[1][:4] #eg: 20180924 to 09/24/2018
        this_study_row['dir_path'] = tag_file.parent.relative_to(data_folder)
        for tag in tag_list:
            this_study_row[tag] = ''

        tag_info.read()
        tag_lines = tag_info.getFileNamesWithTags(tag_list)

        if len(tag_lines)==0:
            print('Skipping, No requested tags found in csv file {}'.format(tag_file))
        for tag_line in tag_lines:
            tag = tag_line['tag']
            tag_path = tag_line['File']
            this_study_row[tag] = tag_path

        study_rows.append(this_study_row)

    if len(study_rows) > 0:

        try:
            csvwrap.CSVWrap.writeCSV(study_rows, Path(args.out_csv))
            print('-------DONE--------')
        except Exception as e:
            print('Error writing output file: \n {}'.format(e))

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with study folders that have an info_corrected.csv file generated', required=True)
    parser.add_argument('--out_csv', type=str, help='Path to the output csv file', required=True)
    parser.add_argument('--tag_list', type=str, help=' Space delmited list of tags to be extracted')

    args = parser.parse_args()

    main(args)