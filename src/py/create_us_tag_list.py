
from pathlib import Path
import preprocess_us
import pytesseract
import csv
from matplotlib import pyplot as plt
import SimpleITK as sitk
import numpy as np
import pprint

def getTesseractTag(np_array, tag_list):
    """
    Run tesseract on an input image.
    np_array: is supposed to be a numpy 2d array
    tag_list: is a list of strings acceptable as tags
    NOTE/TODO: spaces are ignored in the tag list right now. 
    So, POST PLAC, ANT PLAC, etc are not recognized yet.
    """
    config_file = '--oem 1 --psm 12'
    data = pytesseract.image_to_data(np_array, output_type= pytesseract.Output.DICT, config=config_file)
    print(data)
    conf = list(map(int, data['conf']))
    final_tag = 'Undecided'
    if max(conf) > 0:
        max_conf_tag_ind = conf.index(max(conf))
        tag = (data['text'][max_conf_tag_ind]).upper()
        tag = ''.join(e for e in tag if e.isalnum())
        # does this tag live in the list?
        if tag in tag_list:
            final_tag = tag
    print(final_tag)
    return final_tag

def extractTagFromFrame(np_frame, bounding_box, tag_list):
    """
    Do necessary preprocessing on the numpy array, and pass it to tesseract.
    np_fram: 2d grayscale numpy array
    bounding_box: a list of two lists defining the upper left and lower right corners of
    the bounding box where the tag is estimated to be. NOTE: this is very crucial here/
    tag_list: list of acceptable tags. See the note in getTesseractTag()
    """
    #sub_image = np_frame[bounding_box[0][1] : bounding_box[1][1],
    #                     bounding_box[0][0] : bounding_box[1][0] ]
    Dimension = len(np_frame.shape)
    # check if the input dimension is 2
    if Dimension is not 2:
        print('Error: Expecting a 2d image as an input to extract the tag, got another dimension, returning')
        return None

    sitk_image = sitk.GetImageFromArray(np_frame)
    size = sitk_image.GetSize()
    cropped_image = sitk.Crop(sitk_image, bounding_box[0],
                [size[i] - bounding_box[1][i] for i in range(Dimension)])

    plt.imshow(sitk.GetArrayFromImage(cropped_image), cmap='gray')
    plt.pause(0.5)
    plt.show()

    # create processing configurations
    # This looks like a bit hacky way of doing things,
    # but a bit of experimentatio showed that there's no single way
    # of getting to the tags. 
    # TODO: maybe normalization/histogram equalization would help?
    threshold_method = [0, 1] #0 for binary, 1 for otsu
    scale = [1, 2]
    ballsize = [0, 1, 3]
    process_config = [ (t, s, b) for t in threshold_method 
                                 for s in scale 
                                 for b in ballsize  ]
    
    final_tag = 'Undecided'
    fail_tag_list = ['Undecided', 'No tag']
    # go through each one
    for config in process_config:
        print('Trying config: ')
        print(config)
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
        
        final_tag = getTesseractTag(sitk.GetArrayFromImage(final_image), tag_list)
        if final_tag not in fail_tag_list:
            break
    
    return final_tag



data_folder = Path("/Users/hinashah/UNCFAMLI/Data/")
subjects = ["UNC-0414-1_20190830_111550", 
            "UNC-0414-2_20190924_085140",
            "UNC-0366-3_20190927_093424",
            "UNC-0394-3_20191001_113111",
            "UNC-0418-1_20190904_090711",
            "UNC-0447-1_20190927_114657"]
tag_list_file = 'us_tags.txt'
out_images_dir = data_folder / 'Images_2' 
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
tag_statistic['No tag'] = 0

preprocess_us.checkDir(out_images_dir, False)
#plt.ion()
plt.figure()


for subject_id in subjects:

    print("=========== PROCESSING SUBJECT: {} ===============".format(subject_id))
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
