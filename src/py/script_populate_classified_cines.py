from pathlib import Path 
from argparse import ArgumentParser
import os
import json

if __name__  == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Output directory of the prepare_data_lstm_hc.py', required=True)
    parser.add_argument('--out_json', type=str, help='Path for the output json file that will output all the information')
    args = parser.parse_args()

    data_dir = Path('/HOMER_STOR/hinashah/dataset_C1_HC_fullset/')
    list_cine_folders = list(data_dir.glob('**/cine_class_images'))

    print('Found {} studies with a cine folder'.format(len(list_cine_folders)))
    cine_dict = []
    for cine_folder in list_cine_folders:
        all_files = os.listdir(cine_folder)
        print('Study: {} has  {}  cine frames'.format(cine_folder, len(all_files)))
        cine_dict.append({'study_folder' : str(cine_folder.parent), 'num_cine_images' : str(len(all_files))})

    try:
        j_f = json.dumps(cine_dict, indent=4)
        with open(args.out_json,"w") as f:
            f.write(j_f)
    except Exception as e:
        print(e)

    print('--- DONE ----')
