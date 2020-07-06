from argparse import ArgumentParser
import preprocessus
import utils
from pathlib import Path
import logging
import pydicom
import numpy as np
import SimpleITK as sitk
#import cv2

# def dumpImagesForCineMP4(path_to_video, dump_dir, frame_list=None, verbatim=True):
#     msg=''
#     try:
#         cap = cv2.VideoCapture(str(path_to_video))
#         # Check if camera opened successfully
#         if (cap.isOpened() is False): 
#             msg = "Error opening video stream or file"
#             raise IOError(msg)

#         if frame_list is None:
#             frame_list = range(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         # Read until video is completed
#         success,image = cap.read()
#         frame_num = 0
#         while success:
#             if frame_num in frame_list:
#                 out_frame_path = dump_dir/("frame_{}.jpg".format(frame_num))
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 cv2.imwrite( str(out_frame_path), gray)
#                 frame_list.remove(frame_num)
#                 if len(frame_list)==0:
#                     break
#             # Capture frame-by-frame
#             success, image = cap.read()
#             frame_num+=1    
#         # When everything done, release the video capture object
#         cap.release()
#     except Exception as e:
#         msg = "Error dumping mp4 image frames: {}".format(e)
#         if verbatim:
#             logging.warning(msg)
#     finally:
#         return msg

def dumpImagesForCine(cine_file_path, out_dir, store_jpg = False, frame_list = None, verbatim=True):
    msg = ''
    if not cine_file_path.exists():
        msg =  'PATH: {} does not exist'.format(str(cine_file_path))
        if verbatim:
            logging.debug(msg)
        return None, msg
    out_path = out_dir/cine_file_path.stem
    utils.checkDir(out_path, False)

    # Check if this is a dicom or an mp4 file:
    # if (cine_file_path.suffix).lower() == '.mp4':
    #     msg = dumpImagesForCineMP4(path_to_video=cine_file_path, dump_dir=out_path, frame_list=frame_list)
    #     return out_path, msg


    if (cine_file_path.suffix).lower() not in ['.dcm', '.dicom']:
        msg = 'Unsupported video type during frame dump, returning'
        if verbatim:
            logging.debug(msg)
        return None, msg

    file_str = str(cine_file_path)
    # read the metadata header
    ds = pydicom.read_file(file_str)
    if ds is None:
        msg = 'File: {} Missing DICOM metadata'.format(file_str)
        if verbatim:
            logging.debug(msg)
        return None, msg

    sopclass = ds['0008', '0016'].value
    if sopclass != '1.2.840.10008.5.1.4.1.1.3.1':
        msg = 'File: {} not a cine'.format(file_str)
        if verbatim:
            logging.debug(msg)
        return None, msg

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
    if frame_list is None:
        frame_list = range(num_frames)

    for i in frame_list:
        if(store_jpg):
            file_out = str(out_path) + '/frame_' + str(i) + '.jpg'
            preprocessus.writeImage(np_image[i, :, :], file_out )
        file_out = str(out_path) + '/frame_' + str(i) + '.nrrd'
        sim = sitk.GetImageFromArray(np_image[i,:,:])
        if ['0028','0030'] in ds:
            sim.SetSpacing(ds.PixelSpacing)
        sitk.WriteImage(sim, file_out)
        del sim

    del ds
    del np_image
    return out_path, msg

def main(args):
    data_folder = Path(args.dir)
    out_dir = Path(args.out_dir)

    utils.checkDir(out_dir, False)

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
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Study directory', required=True)
    parser.add_argument('--out_dir', type=str, help='Output folder name', required=True)
    parser.add_argument('--store_jpg', action='store_true', help='Store the JPG images')
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    args = parser.parse_args()

    main(args)