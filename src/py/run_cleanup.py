from argparse import ArgumentParser
import preprocessus
import utils
from pathlib import Path
import logging
import pydicom
import numpy as np
import SimpleITK as sitk

def main(args):
    data_folder = Path(args.dir)
    out_dir = Path(args.out_dir)

    utils.setupLogFile(out_dir, args.debug)
    # Populate the filenames21
    file_names = list( data_folder.glob('**/*.dcm') )
    # For each file, extract the first component
    for file_str in file_names:
        # Make sure path exists
        file_path = Path(file_str)
        if not Path(file_path).exists():
            logging.warning('File: {} does not exist,returning'.format(file_str))

        # Double check that the extension is a dicom
        if file_path.suffix != '.dcm':
            logging.warning('Image {} not a dicom, returning'.format(file_str))

        # read the metadata header
        ds = pydicom.read_file(file_str)
        if ds is None:
            logging.warning('File: {} Missing DICOM metadata'.format(file_str))

        sopclass = ds['0008', '0016'].value
        if sopclass ~= '1.2.840.10008.5.1.4.1.1.3.1':
            continue;

        # Get numpy arra
        np_image = ds.pixel_array
        photometric_interpretation = ds['0028','0004'].value
        if photometric_interpretation in ["YBR_FULL_422", "RGB"]:
            # Grab the Y or R channel of the image.
            # NOTE: for RGB image, it is assumed that the labels are in the "red channel", so grabbing R channel would suffice
            np_image =  np_image[:, :, :, 0]
            ds['0028', '0004'].value = 'MONOCRHOME2'
            ds['0028', '0002'].value = 1

        ds['0028', '2110'].value = '00'
        ds.decompress()

        num_frames = ds['0028', '00008'].value
        out_path = str(out_dir/'imgdump')
        utils.checkDir(out_path)
        for i in range(num_frames):
            file_out = out_path + '/frame_' + str(i) + '.jpg'
            preprocessus.writeImage(np_image[i, :, :], file_out )

        # Run cleanup

        # Run correction
       
       # Combine all code

if __name__=="__main__":
# /Users/hinashah/famli/Groups/Restricted_access_data/Clinical_Data/EPIC/Dataset_B
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with the cines that need to be cleaned of the text')
    parser.add_argument('--out_dir', type=str, help='Output folder name')
    parser.add_argument('--cleanup_model', type=str, help='Path to the model file for cleaning up text')
    parser.add_argument('--correction_model', type=str, help='Path to the model file for correction of cleaned up images')
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    args = parser.parse_args()

    main(args)