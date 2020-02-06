import pydicom
from pathlib import Path
import logging
import SimpleITK as sitk

file_str = '/Users/hinashah/UNCFAMLI/Data/us_tags_extract/M/1.2.276.0.26.1.1.1.2.2018.303.48934.7667412.180224000.dcm'
file_path = Path(file_str)

# Make sure path exists
if not Path(file_path).exists():
    logging.warning('File: {} does not exist,returning'.format(file_str))

# Double check that the extension is a dicom
if file_path.suffix != '.dcm':
    logging.warning('Image {} not a dicom, returning'.format(file_str))

# read the metadata header
ds = pydicom.read_file(file_str)
if ds is None:
    logging.warning('File: {} Missing DICOM metadata'.format(file_str))

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

#number of frames is:
#dssingle = ds
#dssingle['0028', '00008'].value = 1
out_path = '/Users/hinashah/famli/Users/hinashah/CleanupNN/imgdump'
for i in range(num_frames):
    ithframe = np_image[i, :, :]
 #   dssingle.PixelData = ithframe.tostring()
    file_out = out_path + '/frame_' + str(i) + '.jpg'
    sim = sitk.GetImageFromArray(ithframe)
    sitk.WriteImage(sim, str(file_out), False)



# from pathlib import Path
# import csv


# data_folder = Path('/Volumes/med/Users/hinashah/dataset_C1_tags/')

# missingsr =[]
# singlesr = []
# morethanonesr = []
# tag_file_names = list( data_folder.glob('**/info.csv') )
# print('Found {} studies'.format(len(tag_file_names)))
# for tag_file in tag_file_names:
#     files_to_copy = []
#     try:
#         with open(tag_file) as f:
#             csv_reader = csv.DictReader(f)
#             file_tag_pairs = [ line['File'] for line in csv_reader if line['type'] == 'Comprehensive SR' ]
#             if len(file_tag_pairs) == 0:
#                 missingsr.append(tag_file)
#             elif len(file_tag_pairs) == 1:
#                 singlesr.append(tag_file)
#             else:
#                 morethanonesr.append(tag_file)
#     except (OSError) as e:
#         print('Error reading csv file: {}'.format(tag_file))


# print('There are 0 SRs in: {} studies'.format(len(missingsr)))
# print('There are 1 SRs in: {} studies'.format(len(singlesr)))
# print('There are more than 1 SRs in: {} studies'.format(len(morethanonesr)))


# dest = '/Users/hinashah/UNCFAMLI/Data/StructuredReports/'
# fn = Path(dest, 'C1_MissingSRs.txt')
# with open( str(fn), 'w' ) as f:
#     l = map(lambda x: str(x) + '\n', missingsr)
#     f.writelines(l)

# fn = Path(dest, 'C1_SingleSRs.txt')
# with open( str(fn), 'w' ) as f:
#     l = map(lambda x: str(x) + '\n', singlesr)
#     f.writelines(l)

# fn = Path(dest, 'C1_MultipleSRs.txt')
# with open( str(fn), 'w' ) as f:
#     l = map(lambda x: str(x) + '\n', morethanonesr)
#     f.writelines(l)
