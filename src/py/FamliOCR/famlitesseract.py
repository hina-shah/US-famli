from pathlib import Path
import numpy as np
import pytesseract
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import re


def getTesseractTagLine(np_array, pattern = None, tag_list = None, single_line=False):
    """
    Run tesseract on an input image.
    np_array: is supposed to be a numpy 2d array
    tag_list: is a list of strings acceptable as tags
    NOTE/TODO: spaces are ignored in the tag list right now. 
    So, POST PLAC, ANT PLAC, etc are not recognized yet.
    pattern: is an re expression in UPPER CASE
    """
    
    if single_line:
        config_file = '--oem 1 --psm 7'
    else:
        config_file = '--oem 1 --psm 12'
    data = pytesseract.image_to_data(np_array, output_type= pytesseract.Output.DICT, config=config_file)
    final_tag = ('Undecided', -1)
    if len(data['conf']) == 1 and data['conf'][0] == '-1':
        # print('No text found')
        return final_tag
    
    # print(data)
    conf = list(map(int, data['conf']))
    if max(conf) > 0:
        if tag_list is not None:
            max_conf_tag_ind = conf.index(max(conf))
            tag = (data['text'][max_conf_tag_ind]).upper()
            tag = ''.join(e for e in tag if e.isalnum())
            # does this tag live in the list?
            if tag in tag_list:
                final_tag = (tag, max_conf_tag_ind)
        
        if pattern is not None:
            found_tags = [ (tag, conf) for (tag, conf) in zip(data['text'], data['conf']) if re.match(pattern, tag.upper())  ]
            if len(found_tags) > 1:
                print('----- WARNING found more than one tags for pattern: {}'.format(pattern))
            if len(found_tags) > 0:
                final_tag = found_tags[0]
        
        if pattern is None and tag_list is None:
            max_conf_tag_ind = conf.index(max(conf))
            tag = (data['text'][max_conf_tag_ind])
            final_tag = (tag, max_conf_tag_ind)

    return final_tag

def iterateThroughConfigs(input_image, pattern=None, tag_list=None, single_line=True, debug=False):
    """
    Preprocess the image and get tag extraction.
    pattern: is an RE pattern to match the text extracted. Use single_line=True with this
    tag_list: is list of tags to look for. Sometimes single line helps with this, 
        but for most part single_line=False will work. 
    debug: If debug is true then images will be displayed using matplotlib
    """

    threshold_method = [0, 1] #0 for binary, 1 for otsu
    scale = [1, 2] #sometimes scaling it twice helps to get the tag
    ballsize = [0, 1, 3] # Ball size for binary dilation
    smoothing = [0, 1] # level of smoothing. 1: pixel
    process_config = [ (t, s, b, sm)    for t in threshold_method 
                                    for s in scale 
                                    for b in ballsize 
                                    for sm in smoothing
                                    ]

    final_tag = 'Undecided'
    final_tag_list = []
    conf_list = []

    # go through each one
    for config in process_config:
        if debug:
            print(config)
        if config[3] > 0:
            cropped_image = sitk.DiscreteGaussian(input_image, float(config[3]))

        if config[0] == 0:
            thresholded_image = (input_image < 128)*255
        else:
            thresholded_image = sitk.OtsuThreshold(input_image)*255

        if config[1] == 1:
            expanded_image = thresholded_image
        else:
            expanded_image = sitk.Expand(thresholded_image, [config[1], config[1]], sitk.sitkLinear)
        
        if config[2] == 0:
            final_image = expanded_image
        else: 
            final_image = sitk.BinaryDilate(expanded_image, config[2], sitk.sitkBall, 0., 255.)
        
        if debug:
            plt.imshow(sitk.GetArrayFromImage(final_image), cmap='gray')
            plt.pause(0.5)
            plt.show()

        tag_conf = getTesseractTagLine(sitk.GetArrayFromImage(final_image), pattern=pattern, tag_list=tag_list)
        
        if tag_list is not None and tag_conf[0] in tag_list:
            # When looking for a tag in tag list    
            final_tag = tag_conf[0]
            break
        else:
            final_tag_list.append(tag_conf[0])
            conf_list.append(tag_conf[1])
    
    if tag_list is  None:
        final_tag = final_tag_list [ conf_list.index( max(conf_list) ) ]
    
    del thresholded_image, expanded_image, final_image
    return final_tag


def processBoundingBox(sitk_image, pattern=None, tag_list=None, debug=False):
    
    cropped_image = sitk.RescaleIntensity(sitk_image)

    if debug:
        plt.imshow(sitk.GetArrayFromImage(cropped_image), cmap='gray')
        plt.pause(0.5)
        plt.show()

    use_single_line = pattern is not None and tag_list is None
    final_tag = iterateThroughConfigs(cropped_image, pattern=pattern, tag_list=tag_list, single_line=use_single_line, debug=debug)
    if final_tag == 'Undecided':
        # If the previous line method did not work, re do with switching the mode for single line.
        # The initial selection of using isngle lin ei based on how experimenst on whic method works best for which
        # rype of OCR task. (oattern extraction vs tag extraction)
        if debug:
            print('Trying the single line method: {}'.format(not use_single_line))
        final_tag = iterateThroughConfigs(cropped_image, pattern=pattern, tag_list=tag_list, single_line= (not use_single_line), debug=debug)
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
    tmp_image = sitk.Crop(sitk_image, bounding_box[0],
                [size[i] - bounding_box[1][i] for i in range(Dimension)])
    del sitk_image
    
    final_tag = processBoundingBox(tmp_image, tag_list=tag_list)
    return final_tag    
