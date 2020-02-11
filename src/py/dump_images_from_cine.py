from argparse import ArgumentParser
import preprocessus
import utils
from pathlib import Path
import logging
import pydicom
import numpy as np
import SimpleITK as sitk

def dumpImagesForCine(cine_file_path, out_dir, store_jpg = False):
    if not cine_file_path.exists():
        logging.warning('PATH: {} does not exist'.format(str(cine_file_path)))

    file_str = str(cine_file_path)
    # read the metadata header
    ds = pydicom.read_file(file_str)
    if ds is None:
        logging.warning('File: {} Missing DICOM metadata'.format(file_str))

    sopclass = ds['0008', '0016'].value
    if sopclass != '1.2.840.10008.5.1.4.1.1.3.1':
        logging.warning('File: {} not a cine'.format(file_str) )
        return

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
    out_path = out_dir/cine_file_path.stem
    utils.checkDir(out_path, False)

    for i in range(num_frames):
        if(store_jpg):
            file_out = str(out_path) + '/frame_' + str(i) + '.jpg'
            preprocessus.writeImage(np_image[i, :, :], file_out )
        file_out = str(out_path) + '/frame_' + str(i) + '.nrrd'
        sim = sitk.GetImageFromArray(np_image[i,:,:])
        sim.SetSpacing(ds.PixelSpacing)
        sitk.WriteImage(sim, file_out)
        del sim
    
    del ds
    del np_image
    return out_path

def main(args):
    data_folder = Path(args.dir)
    out_dir = Path(args.out_dir)

    utils.checkDir(out_dir)

    if data_folder.is_dir():
        # Populate the filenames21
        file_names = list( data_folder.glob('**/*.dcm') )
    elif data_folder.suffix == '.dcm':
        file_names = [data_folder]
    else:
        logging.error('Unknown data format, exiting')
        exit(0)

    utils.setupLogFile(out_dir, args.debug)

    # For each file, extract the first component
    for file_path in file_names:
        file_str = str(file_path)
        logging.info('Processing file: {}'.format(file_str));
        # Make sure path exists
        if not Path(file_path).exists():
            logging.warning('File: {} does not exist,returning'.format(file_str))
            continue

        # Double check that the extension is a dicom
        if file_path.suffix != '.dcm':
            logging.warning('Image {} not a dicom, returning'.format(file_str))
            continue

        dumpImagesForCine(file_path, out_dir, args.store_jpg)

    print('---- DONE ----')

if __name__=="__main__":
# /Users/hinashah/famli/Groups/Restricted_access_data/Clinical_Data/EPIC/Dataset_B
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Study directory')
    parser.add_argument('--out_dir', type=str, help='Output folder name')
    parser.add_argument('--store_jpg', action='store_true', help='Store the JPG images')
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    args = parser.parse_args()

    main(args)