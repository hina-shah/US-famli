from pathlib import Path
import os
import csv
from argparse import ArgumentParser
import logging

def main(args):
    data_folder = Path(args.dir)
    info_csv_list = []
    info_csv_list = list( data_folder.glob('**/info_corrected.csv') )

    study_rows = []
    for i,tag_file in enumerate(info_csv_list):
        print('Processing file {}/{}'.format(i, len(info_csv_list)))
        logging.info('--- PROCESSING: {}'.format(tag_file))
        study_name = tag_file.parent.stem
        this_study_row={}
        parts = study_name.split('_')
        # Get the study id
        this_study_row['study_id_studydate'] = study_name # eg: UNC-0026-1_20180924_093918
        this_study_row['study_id'] = parts[0] # eg: UNC-0026-1
        pid = parts[0]
        this_study_row['pid'] = pid[:pid.rfind('-')] # eg: UNC-0026
        # Get the date of the study
        this_study_row['study_date'] = parts[1][4:6] + '/' + parts[1][6:8] + '/' + parts[1][:4] #eg: 20180924 to 09/24/2018
        study_rows.append(this_study_row)

    with open(args.out_csv, 'w') as csv_file:
        field_names = ['study_id_studydate', 'study_id', 'pid', 'study_date']
        csvfilewriter = csv.DictWriter(csv_file, field_names)
        csvfilewriter.writeheader()
        csvfilewriter.writerows(study_rows)
    print('-------DONE--------')

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with study folders that have an info_corrected.csv file generated', required=True)
    parser.add_argument('--out_csv', type=str, help='Path to the output csv file', required=True)
    args = parser.parse_args()

    main(args)