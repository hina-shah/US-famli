
from pathlib import Path
import preprocess_us
import pytesseract
import csv
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
import numpy as np
import pprint

def extractTagFromFrame(np_frame, bounding_box, tag_list):

    sub_image = np_frame[bounding_box[0][1] : bounding_box[1][1],
                         bounding_box[0][0] : bounding_box[1][0] ]
    

    config_file = '--oem 1 --psm 11'
    sub_image = 255-sub_image
    sub_image[sub_image<128] = 0
    sub_image[sub_image>=128] = 255
    img = Image.fromarray(sub_image)  
    img.filter(ImageFilter.BLUR)
    print('max : {}, min: {}'.format(np.amax(sub_image), np.amin(sub_image)))

    data = pytesseract.image_to_data(img, 
                                    output_type= pytesseract.Output.DICT,
                                    config=config_file)
    print(data)

    conf = list(map(int, data['conf']))
    final_tag = 'Undecided'
    if max(conf) > 0:
        max_conf_tag_ind = conf.index(max(conf))
        tag = (data['text'][max_conf_tag_ind]).upper()
        # does this tag live in the list?
        if tag in tag_list:
            final_tag = tag
        elif not tag:
            final_tag = 'No tag'

    print(final_tag)
    return final_tag

data_folder = Path("/Users/hinashah/UNCFAMLI/Data/")
subjects = ["UNC-0414-1_20190830_111550", 
            "UNC-0414-2_20190924_085140",
            "UNC-0366-3_20190927_093424",
            "UNC-0394-3_20191001_113111",
            "UNC-0418-1_20190904_090711",
            "UNC-0447-1_20190927_114657"]
tag_list_file = 'us_tags.txt'
out_images_dir = data_folder / 'Images_1' 
rescale_size = [800,600]

#tag_bounding_box = [[0,60], [200,150]]
# Approximate bounding box of where the tag is written acoording to the 
# us model
tag_bounding_box = { 'V830':[[40,75], [255,190]],
                    'LOGIQe':  [[0,55], [200,160]]}

# list of ultrasound image types whose tags we do not care about right now.
non_tag_us = ['Unknown', 'ge kretz image', 'Secondary capture image report',
                'Comprehensive SR', '3D Dicom Volume']

# read the list of acceptable tags in the ultrasound file
tag_list = None
try:
    with open(Path(__file__).parent / tag_list_file) as f:
        tag_list = f.read().splitlines()
except:
    print('ERROR READING THE TAG FILE')
    tag_list = ['M', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'R15', 'R45', 
                'L15', 'L45', 'L0', 'L1', 'M0', 'M1', 'RTA', 'RTB', 'RTC']

tag_statistic = dict.fromkeys(tag_list, 0)
tag_statistic['Unknown'] = 0
tag_statistic['Undecided'] = 0

preprocess_us.checkDir(out_images_dir, False)
for subject_id in subjects:

    out_dir = out_images_dir / (subject_id + "_images")
    preprocess_us.checkDir(out_dir)
    
    csv_file = open(str(out_dir/ 'info.csv'), 'w')
    field_names = ['File', 'type', 'tag']
    csvfilewriter = csv.DictWriter(csv_file, field_names)
    csvfilewriter.writeheader()
    
    i=1
    file_names = list( (data_folder / subject_id).glob('**/*.dcm') )
    
    for file_name in file_names:

        print("============ FILE {} ==================== ".format(i))
        print(str(file_name))
        np_frame, us_type, capture_model = preprocess_us.extractImageArrayFromUS(file_name, out_dir, None)
        
        # Extract text from the image
        tag = 'Unknown'
        if us_type not in non_tag_us and capture_model in tag_bounding_box.keys():
            tag = extractTagFromFrame(np_frame, tag_bounding_box[capture_model], tag_list)
        tag_statistic[tag] += 1
        
        if capture_model not in tag_bounding_box.keys():
            print('========= US Model: {} was not found ========='.format(capture_model))

        csvfilewriter.writerow({'File': str(file_name), 'type': us_type, 'tag': tag})

        i+=1
    csv_file.close()

pprint.pprint(tag_statistic)
print('---- DONE ----')