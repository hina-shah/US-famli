import sys
from pathlib import Path
import pydicom
import numpy as np
import os
import SimpleITK as sitk
import shutil
import logging
from PIL import Image,ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def writeImage(np_image, out_path):
    sim = sitk.GetImageFromArray(np_image)
    sitk.WriteImage(sim, str(out_path), False)
    del sim

def extractUSCineFrame(ds, out_path, out_size):   
   
    # Get numpy arra
    np_image = ds.pixel_array

    # Get a mid-cine frame
    middle_frame = int(ds['0028', '00008'].value/2)

    # Maybe cine not of value if only one frame exists, return
    if middle_frame == 0:
        logging.warning('Only one frame in the cine, not using it')
        return None

    logging.debug('grabbing middle frame: {}'.format(middle_frame))
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
        logging.warning('UNSUPPORTED PHOTOMETRIC INTERPRETATION: ' + photometric_interpretation)
    del np_image

    if frame is None:   
        return None

    if out_path is not None:
        writeImage(frame, out_path)

    return frame
    

def extractUSImage(file_path, ds, out_path, out_size):    

    photometric_interpretation = ds['0028','0004'].value
    if photometric_interpretation == "RGB":
        #TODO: ideally shouldn't use reading of the file, but use ds.pixel_array
        # However, converting the numpy array to itk image creates itkImageUC3
        # which is inconsistent with the itkRGBUC2 that the grayscale image filter
        # is expecting..
        np_image = np.dot(ds.pixel_array, [0.2989, 0.5870, 0.1140])
        # ImageType = itk.Image[itk.RGBPixel[itk.UC3],3]
        # itk_image = itk.PyBuffer[ImageType].GetImageFromArray(np.ascontiguousarray(np_image))
        # lumfilter = itk.RGBToLuminanceImageFilter.New(Input = itk_image)
        # #itk_image = itk.imread(str(file_path))
        # #lumfilter = itk.RGBToLuminanceImageFilter.New(Input=itk_image)
        # itk_image = lumfilter.GetOutput()
        # # Turn the image to to numpy array to squeeze the extra dimension
        # # TODO: how do I squeeze a hanging dimension in ITK itself?    
        # np_image = itk.GetArrayFromImage(itk_image)
        
        np_image = np.squeeze(np_image.astype(np.uint8))
        #del itk_image
    elif photometric_interpretation == "YBR_FULL_422":
        np_image = np.squeeze(ds.pixel_array[:,:,0])
    elif photometric_interpretation == "MONOCHROME2":
        # Get numpy array
        np_image = np.squeeze(ds.pixel_array)
    else:
        logging.warning('UNSUPPORTED PHOTOMETRIC INTERPRETATION: ' + photometric_interpretation)
        return None
        
    if out_path is not None:
        writeImage(np_image, out_path)

    return np_image
    
def checkDir(dir_path, delete=True):
    # iF the output directory exists, delete it if requested
    if delete is True and dir_path.is_dir():
        shutil.rmtree(dir_path)
    
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)

def extractImageArrayFromUS(file_path, out_dir, rescale_size):

    file_str = str(file_path)

    # Make sure path exists
    if not Path(file_path).exists():
        logging.warning('File: {} does not exist,returning'.format(file_str))
        return None

    # Create the output path
    out_path = None
    if out_dir is not None:
        out_path = out_dir / (file_path.stem + '.jpg')

    # Double check that the extension is a dicom
    if file_path.suffix != '.dcm':
        logging.warning('Image {} not a dicom, returning'.format(file_str))
        return None

    # read the metadata header
    ds = pydicom.read_file(file_str)
    if ds is None:
        logging.warning('File: {} Missing DICOM metadata'.format(file_str))
        return None
    
    np_frame = None
    us_type = 'Unknown'
    sopclass = ds['0008', '0016'].value
    if sopclass == '1.2.840.10008.5.1.4.1.1.3.1':
        # cine images
        logging.debug('processing as a cine')
        np_frame = extractUSCineFrame(ds, out_path, rescale_size)
        us_type = 'cine'
    elif sopclass == '1.2.840.10008.5.1.4.1.1.6.1':
        # See if if's a GE Kretz volume:
        if ['7fe1', '0011'] not in ds:
            # Not a Kretz volume, continue
            logging.debug('processing as an image')
            np_frame = extractUSImage(file_path, ds, out_path, rescale_size)
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
        logging.debug('Unseen sopclass: {}'.format(sopclass))
        # TODO: add sop classes for structured report and 3d volumetric images
        pass
    
    us_model = ''
    if ['0008', '1090'] in ds:
        us_model = ds['0008', '1090'].value
    
    # cleanup
    del ds

    return np_frame, us_type, us_model
    