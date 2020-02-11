from pathlib import Path
from argparse import ArgumentParser
import logging
import os
import csv
import sys
import utils


def main(args):
    data_folder = Path(args.dir)
    out_folder = Path(args.out_dir)

    #  Setup logging:
    utils.setupLogFile(out_folder, args.debug)
    
    none_tags = ['Unknown', 'Undecided', 'No tag']
    
    # Find all the info.csv files:
    tag_file_names = list( data_folder.glob('**/info.csv') )
    for tag_file in tag_file_names:
        logging.info('--- PROCESSING: {}'.format(tag_file))
        # Read the tags
        this_none_tags = []
        file_tag_pairs = []
        try:
            with open(tag_file) as f:
                csv_reader = csv.DictReader(f)
                file_tag_pairs = [ (line['File'], line['type'], line['tag']) for line in csv_reader]
                this_none_tags = [pair[2] for pair in file_tag_pairs if pair[2] in none_tags]
        except (OSError) as e:
            logging.error('Error reading csv file: {}'.format(tag_file))
            continue
        
        if len(file_tag_pairs) == 0:
            logging.warning('Did not find any information in: {}'.format(str(tag_file)))
            continue

        if len(file_tag_pairs) == len(this_none_tags):
        # If all are unknown, then delete the info.csv file, we will come back to it at a later 
        # stage
            logging.info('No known tags in: {}'.format(str(tag_file)))
            #tag_file.unlink()
        else:
            # change the mount destination in file names
            csvrows = []
            for pair in file_tag_pairs:
                fn = (pair[0]).replace('/mnt', args.som_home_dir)
                tp = pair[1]
                tg = pair[2]
                csvrows.append({'File':fn, 'type':tp, 'tag':tg})
            logging.info('Correcting to: {}, with number: {}'.format(tag_file, len(csvrows)))
            if len(csvrows) == len(file_tag_pairs):
                dir_name = tag_file.parent
                csv_file = open(str(dir_name/'info_corrected.csv'), 'w')
                field_names = ['File', 'type', 'tag']
                csvfilewriter = csv.DictWriter(csv_file, field_names)
                csvfilewriter.writeheader()
                csvfilewriter.writerows(csvrows) 
                csv_file.close()

    logging.info('----- DONE -----')

if __name__=="__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with subject subfolders containing info.csv tagfiles.'
                'Every lowest level subfolder will be considered as a study')
    parser.add_argument('--out_dir', type=str, help='Output directory location for logging.')
    parser.add_argument('--som_home_dir', type=str, help='SOM server directory, needed to replace a mount position in the file names')
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    args = parser.parse_args()

    main(args)