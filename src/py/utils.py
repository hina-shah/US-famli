import logging
import sys
import os
from pathlib import Path
import time
import shutil
import csv

def setupLogFile(dest_dir_path, debug=False):
     #  Setup logging:
    loglevel = logging.DEBUG if debug else logging.INFO
    log_file_name = "log" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
    logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S',
                         level=loglevel, filename=dest_dir_path/log_file_name)

def checkDir(dir_path, delete=True):
    # iF the output directory exists, delete it if requested
    if delete is True and dir_path.is_dir():
        shutil.rmtree(dir_path)
    
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)

def writeCSVRows(file_path, rows, field_names):
    try:
        with open(file_path, 'w') as csv_file:
            field_names = field_names
            csvfilewriter = csv.DictWriter(csv_file, field_names)
            csvfilewriter.writeheader()
            csvfilewriter.writerows(rows) 
    except OSError:
        logging.error('OS Error occured while writing to file: {}'.format(file_path))
    except:
        logging.error('Error while attempting to write to csv file: {}'.format(file_path))

def getCineTagsList(in_tags_string=None):
    tags=  []
    if not in_tags_string:
        tag_list_file = 'us_cine_tags.txt'
        try:
            # WARNING: If this file moves, the path here should also change.
            with open(Path(__file__).parent / tag_list_file) as f:
                tags = f.read().splitlines()
        except:
            print('ERROR READING THE TAG FILE')
            tags = ['M', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'R15', 'R45', 'R0', 'RO', 'R1',
                    'L15', 'L45', 'L0', 'LO', 'L1', 'M0', 'M1', 'RTA', 'RTB', 'RTC', 
                    '3D1', 'RLQ', 'RUQ', 'LUQ', 'LLQ', 'EQ']
    else:
        tags = (in_tags_string).split()
    return tags

def getTagsList(in_tags_string= None):
    tags=  []
    if not in_tags_string:
        tag_list_file = 'us_tags.txt'
        try:
            # WARNING: If this file moves, the path here should also change.
            with open(Path(__file__).parent / tag_list_file) as f:
                tags = f.read().splitlines()
        except:
            print('ERROR READING THE TAG FILE')
            tags = ['DVP','TCD','BREECH','ML','CEPHALIC','MR','POST PLAC','ANT PLAC','FUND PLAC','LEFT PLAC','RIGHT PLAC',
                    'CERVIX','PREVIA','LOW PLAC','CRL','BPD','HC','AC','FL','TCD',
                    'M','L0','LO','L1','R0','RO','R1','R15','L15','R45','L45',
                    'RTA','RTB','RTC','C1','C2','C3','C4','C5','C6','3D1','LUQ','LLQ','RLQ','RUQ','EQ','FUNDUS']
    else:
        tags = (in_tags_string).split()
    return tags