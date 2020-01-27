
from pathlib import Path
import os
import csv
from argparse import ArgumentParser
from pprint import pprint, pformat
import logging
import sys
import time
import pydicom

def checkDir(dir_path, delete=True):
    # iF the output directory exists, delete it if requested
    if delete is True and dir_path.is_dir():
        shutil.rmtree(dir_path)
    
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)

def summarizeStudy(study, data_folder, out_dir):
    image_statistic = {}
    image_statistic['cine'] = 0
    image_statistic['2d image'] = 0
    image_statistic['Unknown'] = 0
    image_statistic['volume'] = 0
    
    logging.info("=========== PROCESSING SUBJECT: {} ===============".format(study.name))
   
    i=1
    file_names = list( study.glob('**/*.dcm') )
    csvrows=[]
    for file_name in file_names:
        file_str = str(file_name)
        logging.debug("FILE {}: {}".format(i, str(file_name)))
        logging.debug(str(file_name))
         # read the metadata header
        ds = pydicom.read_file(file_str)
        if ds is None:
            logging.warning('File: {} Missing DICOM metadata'.format(file_str))
            return None
        
        np_frame = None
        us_type = 'Unknown'
        sopclass = ds['0008', '0016'].value
        if sopclass == '1.2.840.10008.5.1.4.1.1.3.1':
            # cine images
            us_type = 'cine'
        elif sopclass == '1.2.840.10008.5.1.4.1.1.6.1':
            # See if if's a GE Kretz volume:
            if ['7fe1', '0011'] not in ds:
                # Not a Kretz volume, continue
                us_type = '2d image'
            else:
                us_type = 'ge kretz image'
        elif sopclass == '1.2.840.10008.5.1.4.1.1.7':
            us_type = 'Secondary capture image report'
        elif sopclass == '1.2.840.10008.5.1.4.1.1.88.33':
            us_type = 'Comprehensive SR'
        elif sopclass == '1.2.840.10008.5.1.4.1.1.6.2':
            us_type = 'volume'
        else:
            logging.debug('Unseen sopclass: {}'.format(sopclass))
            # TODO: add sop classes for structured report and 3d volumetric images
            pass
        
        logging.debug("US TYPE {}".format(us_type))
        if us_type not in image_statistic.keys():
            image_statistic[us_type] = 0
        image_statistic[us_type] += 1
        # cleanup
        del ds
        i+=1
        
    return image_statistic

def main(args):
    data_folder = Path(args.dir)
    out_dir = Path(args.out_dir)

    checkDir(out_dir, False)    
    loglevel = logging.DEBUG if args.debug else logging.INFO
    log_file_name = "log" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
    logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S',
                         level=loglevel, filename=out_dir/log_file_name)

    studies = []
    for dirname, dirnames, __ in os.walk(str(data_folder)):
        if len(dirnames) == 0:
            studies.append(Path(dirname))
            
    logging.info('Found {} directories with images '.format(len(studies)))
    print('Found {} directories with images '.format(len(studies)))

    image_statistic = {}
    image_statistic['cine'] = 0
    image_statistic['2d image'] = 0
    image_statistic['Unknown'] = 0
    image_statistic['volume'] = 0

    # Also read in study directories that might have been finished by a previous run - do not want to rerun them again
    finished_study_file =out_dir/'finished_studies.txt'
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
    else:
        cleaned_studies = studies
    del studies

    finished_studies_summary_file = out_dir/'finished_studies_summary.csv'
    if finished_study_file.exists():
        try:
            logging.info('Opening: {}'.format(finished_studies_summary_file))
            with open(finished_studies_summary_file, 'r') as f:
                csv_reader = csv.DictReader(f)
                for line in csv_reader:
                    image_statistic[line['Type']] += int(line['Number'])
        except (OSError, ValueError) as err:
            logging.warning('Error reading previously created info.csv for subject: {}: {}'.format(study, err))
        except:
            logging.warning('Error reading previously created info.csv for subject: {}'.format(study))
            logging.warning('Unknown except while reading csvt: {}'.format(sys.exc_info()[0]))
 
    i=1
    for study in cleaned_studies:
        this_tag_statistic = summarizeStudy(study, data_folder, out_dir)
        logging.info('Finished processing: {}'.format(study))
        for key, value in this_tag_statistic.items():
            if key not in image_statistic.keys():
                image_statistic[key] = 0;
            image_statistic[key] += value
        endstr = "\n" if i%50 == 0 else "."
        print("",end=endstr)
        with open(finished_study_file, "a+") as f:
            f.write(str(study)+os.linesep)
        i+=1
    
    # Write the csv file
    csvrows=[]
    for key, value in image_statistic.items():
        csvrows.append({'Type': str(key), 'Number': value})

    csv_file = open(str(out_dir / 'finished_studies_summary.csv'), 'w')
    field_names = ['Type', 'Number']
    csvfilewriter = csv.DictWriter(csv_file, field_names)
    csvfilewriter.writeheader()
    csvfilewriter.writerows(csvrows) 
    csv_file.close()
    pprint(image_statistic)
    logging.info(pformat(image_statistic))
    logging.info('---- DONE ----')
    print('------DONE-----------')

if __name__=="__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with subject subfolders.'
                'Every lowest level subfolder will be considered as a study')
    parser.add_argument('--out_dir', type=str, help='Output directory location.'
                'The summaries will be output here')
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    args = parser.parse_args()

    main(args)