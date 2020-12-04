import json
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import math
import sys
import scale
import SimpleITK as sitk
import csv

class InfiniteValueError(Exception):
    pass

def getImScale(maxmeas, maxmeasfile, unit_adjust_scale = 0.1):
    if maxmeasfile is None:
        return -1
    # NOTE: Assuming that image spacing is in mm, so converting it cms
    maxmeasimg = str(maxmeasfile).replace('json', 'nrrd')
    img = sitk.ReadImage(maxmeasimg)
    spacing = img.GetSpacing()
    if spacing[0] == spacing[1]:
        scaledmeas = maxmeas * spacing[0] * unit_adjust_scale
    else:
        print('WARNING: Rectangular pixels not handled yet, using max spacing')
        scaledmeas = maxmeas * max(spacing[0], spacing[1]) * unit_adjust_scale
    return scaledmeas

def getMeasurement(fit_description, anatomy_name):
    measurement = None
    try:
        measurement = -1
        if anatomy_name == 'AC':
            if len(fit_description['radius'])==1:
                measurement = scale.circumference_circle(fit_description["radius"])
        elif anatomy_name == 'HC':
            if len(fit_description['radius'])>1:
                measurement = scale.circumference_ellipse(fit_description["radius"])
        elif anatomy_name == 'FL':
            if "min" in fit_description and "max" in fit_description:
                measurement = scale.distance_points(fit_description["min"], fit_description["max"])
                if math.isinf(measurement):
                    raise InfiniteValueError
        elif anatomy_name == 'BPD':
            if len(fit_description['radius'])>1:
                measurement = min(fit_description['radius'])*2
    except KeyError as e:
        print('Error processing the contents fit description dict passed')
    except InfiniteValueError as e:
        print('Got an infinite value, anatomy: {}'.format(anatomy_name))
        measurement = -1
    except Exception as e:
        print("Other error occured: {}".format(e))
    
    return measurement    

def getFitDescription(json_file):
    try:
        with open(json_file, 'r') as f:
            fit_description = json.load(f)
    except OSError as e:
        print('Error loading json file: {}'.format(json_file))
        return None
    return fit_description

def getMaxMeasurement(json_list, anatomy_name):
    """
    Function to get the maximum anatomy measurement from a list of json files
    These json files are generated by shell scripts, currently at /HOMER_STOR/hinashah/predict-sh
    """
    max_measurement_file = None
    max_measurement = -1
    for json_file in json_list:
            fit_description = getFitDescription(json_file)
            if fit_description is None:
                continue
            measurement = getMeasurement(fit_description, anatomy_name)
            if measurement is None:
                print('Error processing file {}'.format(json_file))
                continue

            if measurement > max_measurement:
                max_measurement = measurement
                max_measurement_file = json_file
    
    return max_measurement, max_measurement_file

def getAverateMeasurement(json_list, anatomy_name):
    """
    Function to get the maximum anatomy measurement from a list of json files
    These json files are generated by shell scripts, currently at /HOMER_STOR/hinashah/predict-sh
    """
    av_measurement = 0
    num_measurements = 0
    for json_file in json_list:
            fit_description = getFitDescription(json_file)
            if fit_description is None:
                continue
            measurement = getMeasurement(fit_description, anatomy_name)
            if measurement is None:
                print('Error processing file {}'.format(json_file))
                continue

            av_measurement += measurement
            num_measurements+=1
    
    return av_measurement/float(num_measurements)

def gatherMeasurementsForSingleImages(this_anatomy, json_files, data_dir):
    csv_rows = []
    for fit_description_file in json_files:
        fit_description = getFitDescription(fit_description_file)
        if fit_description is None:
            continue
        measurement = getMeasurement(fit_description, this_anatomy)
        if measurement is None: 
            continue
        scaled_meas = getImScale(measurement, fit_description_file)
        row = {}
        row['imname'] = (fit_description_file.name).replace('.json', '.dcm')
        row['anatomy'] = this_anatomy
        row['meas'] = scaled_meas
        csv_rows.append(row)
    # Write this csv file
    if len(csv_rows) == 0:
        print('WARNING: no rows were generated for anatomy: {}'.format(this_anatomy))
        return

    try:
        csv_file_path = data_dir / (this_anatomy + '_estimated_measures.csv')
        with open(csv_file_path, 'w') as f:
            field_names = csv_rows[0].keys()
            csv_file = csv.DictWriter(f, field_names)
            csv_file.writeheader()
            csv_file.writerows(csv_rows)
    except OSError as e:
        print('ERROR writing csv file for anatomy {}'.format(this_anatomy))

def main(args):
    data_dir = Path(args.dir)
    anatomy_tags = ['HC', 'AC', 'BPD', 'FL']

    all_dirs_list = list(data_dir.glob('**/*_fit'))

    study_db = {}
    for fit_dir in all_dirs_list:
        study_name = fit_dir.parent.stem 
        this_anatomy = (fit_dir.name).split('_')[0]
        print('Study name: {}, this_anatomy: {}'.format(study_name, this_anatomy))
        measurements = list(fit_dir.glob('**/*.json'))
        print(len(measurements))

        if args.create_file_csv:
            gatherMeasurementsForSingleImages(this_anatomy, measurements, data_dir)
            if this_anatomy == 'HC':
                gatherMeasurementsForSingleImages('BPD', measurements, data_dir)
            continue # DO not do anything else, continue with the next folder

        if study_name not in study_db:
            study_db[study_name]={}
            study_db[study_name]['study_id'] = study_name
            study_db[study_name]['max_measure'] = {}
            study_db[study_name]['num_measures'] = {}
            study_db[study_name]['max_measure_img'] = {}
            for tag in anatomy_tags:
                study_db[study_name]['max_measure'][tag] = -1
                study_db[study_name]['num_measures'][tag] = 0
                study_db[study_name]['max_measure_img'][tag] = None

        study_db[study_name]['num_measures'][this_anatomy] = len(measurements)
        maxmeas, maxmeasfile = getMaxMeasurement(measurements, this_anatomy)

        # scale the maxmes based on image's scale
        scaled_maxmeas = getImScale(maxmeas, maxmeasfile)
        study_db[study_name]['max_measure'][this_anatomy] = scaled_maxmeas
        if maxmeasfile is not None:
            study_db[study_name]['max_measure_img'][this_anatomy] = Path(maxmeasfile.name).stem
        else:
            study_db[study_name]['max_measure_img'][this_anatomy] = ''

        # Also process for BPD measurements
        if this_anatomy == 'HC':
            an = 'BPD'
            study_db[study_name]['num_measures'][an] = len(measurements)
            maxmeas, maxmeasfile = getMaxMeasurement(measurements, an)
            scaled_maxmeas = getImScale(maxmeas, maxmeasfile)
            study_db[study_name]['max_measure'][an] = scaled_maxmeas
            if maxmeasfile is not None:
                study_db[study_name]['max_measure_img'][this_anatomy] = Path(maxmeasfile.name).stem
            else:
                study_db[study_name]['max_measure_img'][this_anatomy] = ''

    if args.create_file_csv:
        #All the data already written, nothing to do, exit
        print('---- DONE GATHERING Image based measurements -----')
        return

    print('Found {} studies with fit directories'.format(len(study_db)))
    print('Creating the CSV DB')
    anatomy_tags_csv = []
    meas_csv_rows = []
    for study in study_db:
        row = {}
        row['study_id'] = study
        for anatomy_tag in study_db[study]['max_measure']:
            row[anatomy_tag] = study_db[study]['max_measure'][anatomy_tag]
            if anatomy_tag not in anatomy_tags_csv:
                anatomy_tags_csv.append(anatomy_tag)
        meas_csv_rows.append(row)
    try:
        csv_file_path = data_dir / 'study_estimated_measures.csv'
        with open(csv_file_path, 'w') as f:
            field_names = meas_csv_rows[0].keys()
            csv_file = csv.DictWriter(f, field_names)
            csv_file.writeheader()
            csv_file.writerows(meas_csv_rows)
    except OSError as e:
        print(e)
        print('Error dumping the measurements csv file')

    print('Writing out the study db information')
    try:
        j_f = json.dumps(study_db, indent=4)
        with open(data_dir/'study_estimated_measures.json',"w") as f:
            f.write(j_f)
    except Exception as e:
        print(e)
        print('Error dumping the measurements json file')
    print('---------- DONE DONA DONE DONE ----------')

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with study folders that have prediction.csv files generated', required=True)
    parser.add_argument('--create_file_csv', type=bool, default=False, help='Flag to process each json file as a separate measurement'
                                                'Rather than all jsons in a folder being part of a study. Useful for 2d images')
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    args = parser.parse_args()

    main(args)