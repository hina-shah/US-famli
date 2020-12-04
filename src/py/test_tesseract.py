from PIL import Image
from FamliOCR import famlitesseract
import preprocessus
from matplotlib import pyplot as plt
import SimpleITK as sitk
import numpy as np
import logging
from pathlib import Path
import re
from argparse import ArgumentParser
import csv
import time
import utils

def main(args):
    image_list = []
    if args.dir:
        image_dir = Path(args.dir)
        image_list = list( image_dir.glob('**/*.dcm') )
        if len(image_list) == 0:
            image_list = list( image_dir.glob('**/*.jpg') )
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
                #if np_frame is not None:
                # OR if you want to only process cine:
                if us_type == 'cine' and np_frame is not None: 
                    sitk_image = sitk.GetImageFromArray(np_frame)
                    size = sitk_image.GetSize()
                else:
                    print('Not an image type')
                    continue
            else:
                sitk_image = sitk.ReadImage(str(image_path))
                if sitk_image.GetNumberOfComponentsPerPixel() == 3:
                    # iF image's RGB convert to grayscale
                    nparray = sitk.GetArrayFromImage(sitk_image)
                    gray_image = np.squeeze(nparray[:,:,0])
                    sitk_image = sitk.GetImageFromArray(gray_image)
                size = sitk_image.GetSize()
                
            # These bounding boxes are for the upper right corner
            if size[0] <= 640:
                bounding_box = [[70,0], [210,125]]
            else: #190
                bounding_box = [[40,75], [255,250]]

            # These bounding boxes are for the lower right corner
            # if size[0] == 640: 
                #     bounding_box = [[475,383], [640,480]]
                # else:
                #     bounding_box = [[825,500], [960,720]]


            # get the bounding box
            Dimension = sitk_image.GetDimension()
            # check if the input dimension is 2
            if Dimension is not 2:
                print('Error: Expecting a 2d image as an input to extract the tag, got another dimension, returning')
                continue 

            print('Size: {}'.format(sitk_image.GetSize()))

            tmp_image = sitk.Crop(sitk_image, bounding_box[0],
                [size[i] - bounding_box[1][i] for i in range(Dimension)])
            start = time.time()
            # tag = famlitesseract.processBoundingBox(tmp_image, tag_list=['HC', 'FL', 'AC', 'TCD'], debug=False)

            # TO GRAB ALL tags:
            # tag_list = utils.getTagsList()

            tag = famlitesseract.processBoundingBox(tmp_image, tag_list=['IM', 'ILO', 'IRO', 'IL0', 'IM0'], debug=False)
            end = time.time()
            print('Image path: {}'.format(image_path))
            print('Bounding box: {}'.format(bounding_box))
            print("==== FINAL TAG: {}, took {}s".format(tag, (end - start)))

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with images')
    parser.add_argument('--image', type=str, help='Path to the image')
    parser.add_argument('--out_dir', type=str, help='JPG output path')

    args = parser.parse_args()

    main(args)
    
