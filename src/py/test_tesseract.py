from PIL import Image
import pytesseract
import preprocessus
from matplotlib import pyplot as plt
import SimpleITK as sitk
import numpy as np
import logging
from pathlib import Path
import re
from argparse import ArgumentParser
import csv


def getTesseractTag(np_array, pattern = None, tag_list = None):
    """
    Run tesseract on an input image.
    np_array: is supposed to be a numpy 2d array
    tag_list: is a list of strings acceptable as tags
    NOTE/TODO: spaces are ignored in the tag list right now. 
    So, POST PLAC, ANT PLAC, etc are not recognized yet.
    """
    config_file = '--oem 1 --psm 7'
    data = pytesseract.image_to_data(np_array, output_type= pytesseract.Output.DICT, config=config_file)
    print('Tesseract extraction')
    final_tag = ('Undecided', -1)
    if len(data['conf']) == 1 and data['conf'][0] == '-1':
            print('No text found')
            return final_tag
    
    print('Text: {}'.format(data['text']))
    print('Left: {}'.format(data['left']))
    print('Top : {}'.format(data['top']))
    print('Conf: {}'.format(data['conf']))
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
    #print('Identified text: {}, Confidence: {}'.format(final_tag, max(conf)))
    return final_tag

def processBoundingBox(sitk_image, pattern=None):
    
    cropped_image = sitk.RescaleIntensity(sitk_image)
    plt.imshow(sitk.GetArrayFromImage(cropped_image), cmap='gray')
    plt.pause(0.5)
    plt.show()

    threshold_method = [0, 1] #0 for binary, 1 for otsu
    scale = [1, 2]
    ballsize = [0, 1, 3]
    smoothing = [0, 1, 2]
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
        if config[3] > 0:
            cropped_image = sitk.DiscreteGaussian(cropped_image, float(config[3]))

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
        
        plt.imshow(sitk.GetArrayFromImage(final_image), cmap='gray')
        plt.pause(0.5)
        plt.show()

        tag_conf = getTesseractTag(sitk.GetArrayFromImage(final_image), pattern)
        final_tag_list.append(tag_conf[0])
        conf_list.append(tag_conf[1])
        del thresholded_image, expanded_image, final_image
        final_tag = final_tag_list [ conf_list.index( max(conf_list) ) ]
    return final_tag

def main(args):
    
    image_list = []
    if args.dir:
        image_dir = Path(args.dir)
        image_list = list( image_dir.glob('**/*.DCM') )
    else:
        image_list = [Path(args.image)]

    print('Found {} images.'.format(len(image_list)))
    for image_path in image_list:
        if image_path.exists():
            
            if image_path.suffix.lower() == '.dcm':
                # process as a dcm
                # Read image, make sure it is 2d
                np_frame, us_type, us_model = preprocessus.extractImageArrayFromUS(image_path, Path(args.out_dir))
                print('US TYPE: {}, MODEL: {}'.format(us_type, us_model))
            
                # convert to sitk
                sitk_image = sitk.GetImageFromArray(np_frame)
                size = sitk_image.GetSize()
                if size[0] == 640: 
                    bounding_box = [[475,383], [640,480]]
                else:
                    bounding_box = [[825,500], [960,720]]
            else:
                sitk_image = sitk.ReadImage(str(image_path))
                size = sitk_image.GetSize()
                bounding_box = [[70,0], [210,125]]

            # get the bounding box
            Dimension = sitk_image.GetDimension()
            # check if the input dimension is 2
            if Dimension is not 2:
                print('Error: Expecting a 2d image as an input to extract the tag, got another dimension, returning')
                continue 

            print('Size: {}'.format(sitk_image.GetSize()))

            tmp_image = sitk.Crop(sitk_image, bounding_box[0],
                [size[i] - bounding_box[1][i] for i in range(Dimension)])
            processBoundingBox(tmp_image)

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with images')
    parser.add_argument('--image', type=str, help='Path to the image')
    parser.add_argument('--out_dir', type=str, help='JPG output path')
    args = parser.parse_args()

    main(args)
    
