from pathlib import Path
import logging

import pydicom
import SimpleITK as sitk
import numpy as np

class DCMIO:
    '''
    Class to provide utility functions for dicom file reading, and other operations
    : Get the photometric interpretation
    : Convert to grayscale
    : For videos - get the number of frames
    : Get and save middle frame
    : Get model type
    : Get the type of the dicom images
    '''
    def __init__(self, filepath):

        self.ds = None
        self.sopclass = ''
        self.us_model = ''
        self.us_type = 'Unknown'
        self.photometric_interpretation = ''

        # read filename
        if isinstance(filepath, str):
            self.file_path = Path(filepath)
        elif isinstance(filepath, Path):
            self.file_path = filepath
        else:
            raise TypeError(('DICOM input File path object is of the wrong type'))
        
        # check the suffix
        if self.file_path.suffix.lower() not in ['.dcm', '.dicom']:
            raise TypeError('DICOM input File extension is of the wrong type')

        if not self.file_path.exists():
            raise IOError('File: {} does not exist,returning'.format(self.file_path))

        # read the dicom metaheader
        file_str = str(self.file_path)
        self.ds = pydicom.read_file(file_str)
        if self.ds is None:
            raise IOError('File: {} Missing DICOM metadata'.format(file_str))
        
        # Set various information values
        self.set_dsdata()

    def set_dsdata(self):
        self.sopclass = self.ds['0008', '0016'].value
        self.us_type = 'Unknown'
        if self.sopclass == '1.2.840.10008.5.1.4.1.1.3.1':
            # cine images
            logging.debug('processing as a cine')
            self.us_type = 'cine'
        elif self.sopclass == '1.2.840.10008.5.1.4.1.1.6.1':
            # See if if's a GE Kretz volume:
            if ['7fe1', '0011'] not in self.ds:
                # Not a Kretz volume, continue
                logging.debug('processing as an image')
                self.us_type = '2d image'
            else:
                logging.debug('processing as an ge kretz image')
                self.us_type = 'ge kretz image'
        elif self.sopclass == '1.2.840.10008.5.1.4.1.1.7':
            self.us_type = 'Secondary capture image report'
        elif self.sopclass == '1.2.840.10008.5.1.4.1.1.88.33':
            self.us_type = 'Comprehensive SR'
        elif self.sopclass == '1.2.840.10008.5.1.4.1.1.6.2':
            self.us_type = '3D Dicom Volume'
        else:
            logging.debug('Unseen sopclass: {}'.format(self.sopclass))

        self.us_model = ''
        if ['0008', '1090'] in self.ds:
            self.us_model = self.ds['0008', '1090'].value
        
        if self.us_type == 'cine' and self.us_model=='LOGIQe':
            self.us_model = 'LOGIQeCine'

        self.photometric_interpretation = ''
        if ['0028','0004'] in self.ds: 
            self.photometric_interpretation = self.ds['0028','0004'].value

    # GETTERS
    def get_type(self):
        return self.us_type

    def get_model(self):
        return self.us_model

    def get_photometric_interpretation(self):
        return self.photometric_interpretation

    def get_sopclass(self):
        return self.sopclass

    def check_photometric_interpretation(self):
        if self.photometric_interpretation not in ['RGB', 'YBR_FULL_422', 'MONOCHROME2']:
            logging.warning('UNSUPPORTED PHOTOMETRIC INTERPRETATION: ' + self.photometric_interpretation)
            return False
        else:
            return True

    def cleanup(self):
        del self.ds
    
    # Get numpy array for the image
    def get_gray_nparray(self):
        if self.ds is  None:
            logging.error('DICOM data empty')
            return None
        if not self.check_photometric_interpretation():
            logging.warning('Unsupported photometric interpretation')
            return None

        np_image = self.ds.pixel_array
        if self.us_type == '2d image':
            if self.photometric_interpretation == "RGB":
                np_image = np.dot(np_image, [0.2989, 0.5870, 0.1140])
                np_image = np.squeeze(np_image.astype(np.uint8))
            elif self.photometric_interpretation == "YBR_FULL_422":
                np_image = np.squeeze(np_image[:,:,0])
            elif self.photometric_interpretation == "MONOCHROME2":
                np_image = np.squeeze(np_image)
        elif self.us_type == 'cine':
            if self.photometric_interpretation in ["YBR_FULL_422", "RGB"]:
                # Grab the Y or R channel of the image.
                # NOTE: for RGB image, it is assumed that the labels are in the "yellow channel", so grabbing R channel would suffice
                np_image =  np_image[:, :, :, 0]
        else:
            logging.debug('DICOM type: {}, cant get nparray of this'.format(self.us_type))
            return None
        
        return np_image

    # Convert dicom to gray and output
    def to_gray_and_write(self, out_path):
        
        gray_image = self.get_gray_nparray()
        if self.us_type == 'cine':
            out_spacing = [float(self.ds.PixelSpacing[0]), float(self.ds.PixelSpacing[1]), 1.]
        else:
            out_spacing = [float(self.ds.PixelSpacing[0]), float(self.ds.PixelSpacing[1])]
        
        try:
            sim = sitk.GetImageFromArray(gray_image)
            sim.SetSpacing(out_spacing)
            sitk.WriteImage(sim, out_path)
        except Exception as e:
            logging.error('Error writing the gray file: {} \n E: {}'.format(file_out. e))

    # Get the middle frame
    def get_middle_frame(self):
        if self.ds is None:
            logging.error('No DICOM data')
            return None
        
        if self.us_type != 'cine':
            logging.warning('Cant get a middle frame for a non-cine')
            return None
        
        # Get a mid-cine frame
        middle_frame = int(self.ds['0028', '00008'].value/2)

        # Maybe cine not of value if only one frame exists, return
        if middle_frame == 0:
            logging.warning('Only one frame in the cine, not using it')
            return None
        frame=None
        np_image = self.get_gray_nparray()
        frame = np_image[middle_frame, :, :]

        return frame

    # Get representative frame
    def get_repres_frame(self, out_path=None):
        '''
        This function returns a middle frame for the cine, and 
        the image itself when 2d.

        Doesn't handle volumes yet.

        All the data is returned in grayscale
        '''

        if self.ds is None:
            logging.error('No DICOM data')
            return None

        frame = None
        if self.us_type == 'cine':
            frame = self.get_middle_frame()
        elif self.us_type == '2d image':
            frame = self.get_gray_nparray()
        else:
            logging.warning('Asking frame from the wrong datatype')

        if frame is not None and out_path is not None:
            sim = sitk.GetImageFromArray(frame)
            sitk.WriteImage(sim, str(out_path), False)
            del sim

        return frame