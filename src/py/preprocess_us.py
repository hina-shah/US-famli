#!/usr/bin/env python

import sys
from pathlib import Path
import pydicom
import numpy as np
import os
import itk
import shutil

def rescaleAndWriteImage(np_image, out_path, out_size):
    # Convert back to ITK, and write out.
    itk_image = itk.GetImageFromArray(np.ascontiguousarray(np_image))
    
    # If requested, rescale to output size
    if out_size is not None:
        PixelType = itk.UC
        Dimension = 2

        # Rescale to output size
        ImageType = itk.Image[PixelType, Dimension]
        resamplerType = itk.ResampleImageFilter[ImageType, ImageType]
        resampleFilter = resamplerType.New()
        
        # calculate output spacing
        input_size = itk_image.GetLargestPossibleRegion().GetSize()
        input_spacing = itk_image.GetSpacing()
        output_spacing = [float(input_spacing[i])*float(input_size[i])/float(out_size[i]) for i in [0,1] ]

        # run resample filter
        resampleFilter.SetInput(itk_image)
        resampleFilter.SetSize(out_size);
        resampleFilter.SetOutputSpacing(output_spacing)
        itk_image = resampleFilter.GetOutput()

    if out_path is not None:
        itk.imwrite(itk_image, str(out_path))
    
    return itk.GetArrayFromImage(itk_image)

def extractUSCineFrame(file_path, ds, out_path, out_size):   
    # Make sure path exists
    if not Path(file_path).exists():
        return

    # Get numpy array
    itk_image = itk.imread(file_path)
    np_image = itk.GetArrayFromImage(itk_image)

    # Get a mid-cine frame
    middle_frame = int(ds['0028', '00008'].value/2)
    print('grabbing middle frame: {}'.format(middle_frame))
    frame = None
    # Depending on the photometric interpretation, grab the first frame and save as jpg
    photometric_interpretation = ds['0028','0004'].value
    if photometric_interpretation == "MONOCHROME2":
        # grab the middle frame
        frame = np_image[middle_frame, :,:]
    elif photometric_interpretation in ["YBR_FULL_422", "RGB"]:
        # Grab the Y or R channel of the image.
        # NOTE: for RGB image, it is assumed that the labels are in the "yellow channel", so grabbing R channel would suffice
        frame =  np_image[middle_frame, :, :, 0]
    else:
        print('UNSUPPORTED PHOTOMETRIC INTERPRETATION: ' + photometric_interpretation)
    if frame is None:   
        return None

    return rescaleAndWriteImage(frame, out_path, out_size)
    

def extractUSImage(file_path, ds, out_path, out_size):    
     # Make sure path exists
    if not Path(file_path).exists():
        return

    itk_image = itk.imread(file_path)
    photometric_interpretation = ds['0028','0004'].value
    if photometric_interpretation == "RGB":
        lumfilter = itk.RGBToLuminanceImageFilter.New(Input=itk_image)
        itk_image = lumfilter.GetOutput()
    
    # Turn the image to to numpy array to squeeze the extra dimension
    # TODO: how do I squeeze a hanging dimension in ITK itself?    
    np_image = itk.GetArrayFromImage(itk_image)
    np_image = np.squeeze(np_image)
    
    return rescaleAndWriteImage(np_image, out_path, out_size)
    
def checkDir(dir_path, delete=True):
    # iF the output directory exists, delete it if requested
    if delete is True and dir_path.is_dir():
        shutil.rmtree(dir_path)
    
    if not dir_path.is_dir():
        dir_path.mkdir()

def extractImageArrayFromUS(file_path, out_dir, rescale_size):
    # Make sure path exists
    if not Path(file_path).exists():
        print('File: {} does not exist,returning'.format(str(file_path)))
        return None

    # Create the output path
    out_path = None
    if out_dir is not None:
        out_path = out_dir / (file_path.stem + '.jpg')

    # Double check that the extension is a dicom
    if file_path.suffix != '.dcm':
        print('Image not a dicom, returning')
        return None

    # read the metadata header
    file_str = str(file_path)
    ds = pydicom.read_file(file_str)
    if ds is None:
        print('Missing DICOM metadata')
        return None
    
    np_frame = None
    us_type = 'Unknown'
    sopclass = ds['0008', '0016'].value
    if sopclass == '1.2.840.10008.5.1.4.1.1.3.1':
        # cine images
        print('processing as a cine')
        np_frame = extractUSCineFrame(file_str, ds, out_path, rescale_size)
        us_type = 'cine'
    elif sopclass == '1.2.840.10008.5.1.4.1.1.6.1':
        # See if if's a GE Kretz volume:
        if ['7fe1', '0011'] not in ds:
            # Not a Kretz volume, continue
            print('processing as an image')
            np_frame = extractUSImage(file_str, ds, out_path, rescale_size)
            us_type = '2d image'
        else:
            us_type = 'ge kretz image'
    elif sopclass == '1.2.840.10008.5.1.4.1.1.7':
        us_type = 'Secondary capture image report'
    elif sopclass == '1.2.840.10008.5.1.4.1.1.88.33':
        us_type = 'Comprehensive SR'
    elif sopclass == '1.2.840.10008.5.1.4.1.1.6.2':
        us_type = '3D Dicom Volume'
    else:
        print('Unseen sopclass: {}'.format(sopclass))
        # TODO: add sop classes for structured report and 3d volumetric images
        pass
    
    us_model = ''
    if ['0008', '1090'] in ds:
        us_model = ds['0008', '1090'].value
    
    return np_frame, us_type, us_model
