import pytesseract
from matplotlib import pyplot as plt
import SimpleITK as sitk
import numpy as np
import logging

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
    logging.debug(data)
    conf = list(map(int, data['conf']))
    final_tag = 'Undecided'
    if max(conf) > 0:
        max_conf_tag_ind = conf.index(max(conf))
        tag = (data['text'][max_conf_tag_ind]).upper()
        tag = ''.join(e for e in tag if e.isalnum())
        # does this tag live in the list?
        if tag in tag_list:
            final_tag = tag
    logging.debug(final_tag)
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
        logging.warning('Error: Expecting a 2d image as an input to extract the tag, got another dimension, returning')
        return None

    sitk_image = sitk.GetImageFromArray(np_frame)
    size = sitk_image.GetSize()
    cropped_image = sitk.Crop(sitk_image, bounding_box[0],
                [size[i] - bounding_box[1][i] for i in range(Dimension)])

    # plt.imshow(sitk.GetArrayFromImage(cropped_image), cmap='gray')
    # plt.pause(0.5)
    # plt.show()

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
