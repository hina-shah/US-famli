from PIL import Image
import pytesseract
#from matplotlib import pyplot as plt
import SimpleITK as sitk
import numpy as np
import logging
from pathlib import Path
import re
from argparse import ArgumentParser
import csv

#max_conf_tag_ind = conf.index(max(conf))
        #tag = (data['text'][max_conf_tag_ind]).upper()
        #tag = ''.join(e for e in tag if e.isalnum())
        # does this tag live in the list?
        #final_tag = tag

# Patterns:
# Gestation Age: r"^(GA=\d\d?W[0-7]D)"
# Study ID: r"^(VIL|UNC)-"
# Depth: r"\d+\.\d*CM"
# Obesity: r"D\s?\d+\.\d*CM"
# Gain: r"-?\d+\.?\d*"

def getTesseractTag(np_array, pattern):
    """
    Run tesseract on an input image.
    np_array: is supposed to be a numpy 2d array
    tag_list: is a list of strings acceptable as tags
    NOTE/TODO: spaces are ignored in the tag list right now. 
    So, POST PLAC, ANT PLAC, etc are not recognized yet.
    """
    config_file = '--oem 1 --psm 7'
    data = pytesseract.image_to_data(np_array, output_type= pytesseract.Output.DICT, config=config_file)
    #print(data)
    conf = list(map(int, data['conf']))
    final_tag = ('Undecided', -1)
    if max(conf) > 0:
        found_tags = [ (tag, conf) for (tag, conf) in zip(data['text'], data['conf']) if re.match(pattern, tag.upper())  ]
        if len(found_tags) > 1:
            print('----- WARNING found more than one tags for pattern: {}'.format(pattern))
        if len(found_tags) > 0:
            final_tag = found_tags[0]
    #print('Identified text: {}, Confidence: {}'.format(final_tag, max(conf)))
    return final_tag

def processBoundingBox(sitk_image, pattern=None):
    
    cropped_image = sitk.RescaleIntensity(sitk_image)
    #plt.imshow(sitk.GetArrayFromImage(cropped_image), cmap='gray')
    #plt.pause(0.5)
    #plt.show()

    threshold_method = [0, 1] #0 for binary, 1 for otsu
    scale = [1, 2]
    ballsize = [0, 1, 3]
    process_config = [ (t, s, b) for t in threshold_method 
                                    for s in scale 
                                    for b in ballsize  ]

    final_tag = 'Undecided'
    final_tag_list = []
    conf_list = []

    # go through each one
    for config in process_config:
        if config[0] == 0:
            thresholded_image = (cropped_image < 128)*255
        else:
            thresholded_image = sitk.OtsuThreshold(cropped_image)*255

        if config[1] == 1:
            expanded_image = thresholded_image
        else:
            expanded_image = sitk.Expand(thresholded_image, [config[1], config[1]], sitk.sitkLinear)
        
        if config[2] == 0:
            final_image = expanded_image
        else: 
            final_image = sitk.BinaryDilate(expanded_image, config[2], sitk.sitkBall, 0., 255.)
        
        tag_conf = getTesseractTag(sitk.GetArrayFromImage(final_image), pattern)
        final_tag_list.append(tag_conf[0])
        conf_list.append(tag_conf[1])
        del thresholded_image, expanded_image, final_image
        final_tag = final_tag_list [ conf_list.index( max(conf_list) ) ]
    return final_tag

def processImage(img_path, find_obesity=False):
    sitk_image = sitk.ReadImage(str(img_path))
    Dimension = sitk_image.GetDimension()
    # check if the input dimension is 2
    if Dimension is not 2:
        print('Error: Expecting a 2d image as an input to extract the tag, got another dimension, returning')
        return

    tag_bounding_box = { 'Depth':[[395,45], [470,70], r"\d+\.\d*CM"],
                     'Gain':  [[940,120], [960,137], r"-?\d+\.?\d*" ],
                     'GA': [[65,45], [390,70], r"^(GA=\d\d?W[0-7]D)"],
                     'Obesity': [[865, 690], [960,715], r"D\s?\d+\.\d*CM"]
                    }
    text_dict = {}
    fail_tag_list = ['Undecided', 'No tag']
    size = sitk_image.GetSize()
    for tag in tag_bounding_box:
        text_dict[tag] = ''
        if tag=='Obesity' and not find_obesity:
            continue
        print(' --- Processing for {}'.format(tag))
        bounding_box = tag_bounding_box[tag]
        tmp_image = sitk.Crop(sitk_image, bounding_box[0],
                    [size[i] - bounding_box[1][i] for i in range(Dimension)])

        final_tag = processBoundingBox(tmp_image, bounding_box[-1] )
        del tmp_image
        print('--- Final tag for object {} is: {}'.format(tag, final_tag))
        if final_tag in fail_tag_list:
            continue
        text_dict[tag] = final_tag

    return text_dict

def postProcess(created_csv, out_csv):

    in_csv_path = Path(created_csv)
    out_csv_path = Path(out_csv)
    try:
        out_csv_rows =[]
        with open(in_csv_path, 'r') as f:
            csvreader = csv.DictReader(f)
            processed_studies = set()
            for line in csvreader:
                processed_studies.add(line['StudyID'])
                out_row = {}
                out_row['StudyID'] = line['StudyID']
                out_row['Tag'] = line['Tag']
                out_row['Depth'] = line['Depth']
                out_row['GA'] = line['GA']
                out_row['Obesity'] = line['Obesity']
                out_row['Gain'] = line['Gain']
                out_row['DepthNum'] = out_row['Depth'][:out_row['Depth'].find('cm')] if len(out_row['Depth']) > 0 else ''
                ga_str = out_row['GA'].upper()
                if len(ga_str) > 0:
                    ga_w = int(ga_str[ga_str.find('=')+1:ga_str.find('W')])
                    ga_d = int(ga_str[ga_str.find('W')+1:ga_str.find('D')])
                    ga = ga_w*7+ga_d 
                    ga_str = str(ga)
                out_row['GANum']  = ga_str
                obesity = out_row['Obesity'].upper()
                if len(obesity) > 0:
                    obesity = obesity[ obesity.find('D')+1:obesity.find('CM') ]
                out_row['ObesityNum'] = obesity
                out_csv_rows.append(out_row)

        print('----- PROCESSED STUDIES: {}'.format(len( set(processed_studies))))
        print('----- number of rows: {}'.format(len(out_csv_rows)))
        
        if len(out_csv_rows) > 0:
            with open(out_csv_path, 'w', newline='') as f:
                headers =  out_csv_rows[0].keys()
                csvwriter = csv.DictWriter(f, headers) 
                csvwriter.writeheader()
                csvwriter.writerows(out_csv_rows)

    except Exception as e:
        print('Error Post Processing the created file: {}'.format(e))

def main(args):

    if args.post_proc and args.post_proc_csv:
        print('Running post processing')
        postProcess(args.post_proc_csv, args.out_csv)
        print('----- DONE -----')
        return

    # Uncomment the following line if the tesseract is not in command line path
    #pytesseract.pytesseract.tesseract_cmd = r'P:/Software/Tesseract/tesseract'

    dir_path = Path(args.dir)
    out_csv_path = Path(args.out_csv)
    out_dir = out_csv_path.parent
    
    # Find the tags in a file. 
    tag_list_file = 'us_cine_tags.txt'
    try:
        with open(Path(__file__).parent / tag_list_file) as f:
            tag_list = f.read().splitlines()
    except:
        print('ERROR READING THE TAG FILE')
        tag_list = ['M', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'R15', 'R45', 'R0', 'RO', 'R1',
                        'L15', 'L45', 'L0', 'LO', 'L1', 'M0', 'M1', 'RTA', 'RTB', 'RTC']
    tag_list.append('CERVIX')    

    # Assuming that the input directory has the diretory structure with info_corrected.csv files
    csv_list = list(dir_path.glob('**/info_corrected.csv'))

    # If the output csv arleady exists, we will skip the studies that were already processed.
    # If you want to start over, delete the output file. 
    processed_studies = []
    try:
        if out_csv_path.exists():
            with open(out_csv_path, 'r') as f:
                csvreader = csv.DictReader(f)
                processed_studies = [line['StudyID'] for line in csvreader]
                processed_studies_set = set(processed_studies)
                info_csv_list = [ csv_file for csv_file in csv_list if (csv_file.parent).stem not in processed_studies_set ]
                print('----- PROCESSED STUDIES: {}'.format(len(processed_studies_set)))
                print('------ REMAININST STUDIES: {}'.format(len(info_csv_list)))
        else:
            info_csv_list = csv_list
            print('Processed studies file does not exist, starting over.')
    except Exception as e:
        print('--- Error reading the already processed studies, starting over')

    #csv_rows=[]
    with open(str(out_csv_path), 'a', newline='') as f:
        field_names = ['StudyID', 'Tag', 'Depth', 'Gain', 'GA', 'Obesity']
        out_csv_writer = csv.DictWriter(f, field_names)
        if len(processed_studies) == 0:
            out_csv_writer.writeheader()
        #cnt = 5
        for i,csv_file in enumerate(info_csv_list):
            print('Processing file {}/{}'.format(i, len(info_csv_list)))
            study_id = (csv_file.parent).stem
            if study_id in processed_studies:
                # skipping
                print('Found study id {} in output file, Skipping'.format(study_id))
                continue

            # Case not processed already. run OCR.
            try:
                with open(csv_file, 'r') as f:
                    csv_reader = csv.DictReader(f)
                    tag_lines = [line for line in csv_reader if line['tag'] in tag_list]
                msg = 'Found: {} tag lines'.format(len(tag_lines)) + '\n'
                for line in tag_lines:
                    file_path = Path(line['File'])
                    path_jpg = csv_file.parent / (file_path.stem + '.jpg')
                    extract_obesity = (line['tag'] == 'CERVIX')
                    if extract_obesity is True:
                        print('Should find cervix')
                    text_dict = processImage(path_jpg, extract_obesity)
                    text_dict['Tag'] = line['tag']
                    text_dict['StudyID'] = (csv_file.parent).stem
                    print('GOT',text_dict)
                    #csv_rows.append(text_dict)
                    out_csv_writer.writerow(text_dict)
                #cnt -=1 
                #if cnt == 0:
                #    break      
            except Exception as e:
                print('ERROR processing the csv file: {} \n {}'.format(csv_file, e))

    # if len(csv_rows) > 0:
    #     with open(str(out_csv_path), 'w', newline='') as f:
    #         field_names = csv_rows[0].keys()
    #         csvfilewriter = csv.DictWriter(f, field_names)
    #         csvfilewriter.writeheader()
    #         csvfilewriter.writerows(csv_rows) 

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with study folders that have an info_corrected.csv file generated')
    parser.add_argument('--post_proc_csv', type=str, help='the CSV file that was created by a previous run and needs to be postprocessed')
    parser.add_argument('--out_csv', type=str, help='Path to the output csv', required=True)
    parser.add_argument('--post_proc', type=bool, default = False, help='Postprocess the already created csv')
    args = parser.parse_args()

    main(args)
    