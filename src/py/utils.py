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
