from argparse import ArgumentParser
from pathlib import Path
import utils
import logging
import csv

def main(args):
    data_folder = Path(args.dir)
    
    anatomies_folder = Path(args.anatomies_dir)
    pair_folder_names = ['Pair1', 'Pair2']

    out_images_dir =Path(args.out_dir)
    utils.setupLogFile(out_images_dir, True)

    # read the ga and biom measurements table
    ga_db = {}
    try:
        with open(args.ga_measurements_db, 'r') as f:
            csv_reader = csv.DictReader(f)
            for line in csv_reader:
                ga_db[line['study_id_studydate']] = line
    except OSError as e:
        print('Error in reading the DB file {}'.format(args.ga_measurements_db))
        return

    info_csv_list = []
    info_csv_list = list( data_folder.glob('**/info_corrected.csv') )

    image_2d_tags=['AC','HC', 'FL']
    anatomy_csvs = {}
    anatomy_ga_counts = {}
    anatomy_meas_counts = {}
    anatomy_both_counts = {}
    for tag in image_2d_tags:
        anatomy_csvs[tag] = []
        anatomy_ga_counts[tag] = [0, 0]
        anatomy_meas_counts[tag] = [0, 0]
        anatomy_both_counts[tag] = [0, 0]

    study_rows = []
    for i,tag_file in enumerate(info_csv_list):
        print('Processing file {}/{}'.format(i, len(info_csv_list)))
        logging.info('--- PROCESSING: {}'.format(tag_file))
        # Find the lines with the tags above and if they are 2d images
        try:
            with open(tag_file) as f:
                csv_reader = csv.DictReader(f)
                image_list = [ [line['tag'], line['File']] for line in csv_reader if line['tag'] in image_2d_tags and line['type'] == '2d image']
                study_name = tag_file.parent.stem
                logging.info("# of images: {}".format(len(image_list)))
                for image_pair in image_list:
                    # See if the file is in either Pair0 or Pair1 of the corresponding anatomy
                    file_path = Path(image_pair[1])
                    tag = image_pair[0]
                    file_name = file_path.name 
                    which_dir = None
                    
                    for pair in pair_folder_names:
                        if (anatomies_folder/tag/pair/ (file_path.stem + '.jpg')).exists():
                            which_dir = pair
                            break
                    if which_dir is not None and study_name in ga_db:
                        # Create a row in the anatomy folder for the tag with the following fields:
                        file_row = {}
                        # Full study id
                        file_row['study_name'] = study_name
                        # Which subfolder iin anatomies folder
                        file_row['pair_folder'] = which_dir
                        # The anatomy
                        file_row['anatomy'] = tag
                        # Image name
                        file_row['image_name'] = file_path.name
                        
                        ga_db_entry = ga_db[study_name]
                        # Corresponding anatomy value and ga
                        file_row['ga'] = int(ga_db_entry['ga_edd']) if ga_db_entry['ga_edd'] != "." else -1
                        file_row['meas'] = float(ga_db_entry[tag.lower() + '_1']) if ga_db_entry[ tag.lower() + '_1'] != "." else -1
                        # Image path
                        file_row['original_path'] = file_path
                        anatomy_csvs[tag].append(file_row)
                        anatomy_ga_counts[tag][which_dir == pair_folder_names[1]] += file_row['ga'] > 0
                        anatomy_meas_counts[tag][which_dir == pair_folder_names[1]] += file_row['meas'] > 0
                        anatomy_both_counts[tag][which_dir == pair_folder_names[1]] += file_row['ga'] > 0 and file_row['meas'] > 0
                    else:
                        logging.info(" Image {}, tag: {}, which_pair: {}, in ga db?: {} ".format(file_path, tag, which_dir, study_name in ga_db))

        except (OSError) as e:
            logging.error('Error reading csv file: {}'.format(tag_file))
            return
        except KeyError as e:
            logging.error('Error in processing the keys in a dict {}'.format(e))
        except Exception as e:
            logging.error('Error in processing the tag file: {}'.format(e))

    # Write out all the csvs
    for tag in image_2d_tags:
        headers = ['study_name', 'anatomy', 'pair_folder', 'image_name', 'ga', 'meas', 'original_path']
        file_path = out_images_dir / (tag + '.csv')
        with open(file_path, 'w') as csv_file:
            csvfilewriter = csv.DictWriter(csv_file, headers)
            csvfilewriter.writeheader()
            csvfilewriter.writerows(anatomy_csvs[tag])
    
    print(anatomy_ga_counts)
    print(anatomy_meas_counts)
    print(anatomy_both_counts)
    logging.info("# of  images with valid GA in [Pair1, Pair2] folders: \n {}".format(anatomy_ga_counts))
    logging.info("# of  images with valid Measurements in [Pair1, Pair2] folders: \n {}".format(anatomy_meas_counts))
    logging.info("# of  images with bot GA and Measurements in [Pair1, Pair2] folders: \n {}".format(anatomy_both_counts))

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with study folders that have an info_corrected.csv file generated', required=True)
    parser.add_argument('--out_dir', type=str, help='Output directory location.', required=True)
    parser.add_argument('--anatomies_dir', type=str, help='Direcotry that has anatomy images separated into caliper and non-caliper images', required=True)
    parser.add_argument('--ga_measurements_db', type=str, help='CSV file generated from C1_Main.sas that has studies and their respective anatomy measurements and gas')
    args = parser.parse_args()

    main(args)