from pathlib import Path
from argparse import ArgumentParser
import run_segmentation_measurement
import utils 
import multiprocessing as mp
import csv
import shutil
import SimpleITK as sitk
import pydicom

def dcm2nrrd(file_name, target_file_name):
    file_str = str(file_name)
    # read the metadata header
    ds = pydicom.read_file(file_str)
    if ds is None:
        print('File: {} Missing DICOM metadata'.format(file_str))
        return
    # Get numpy arra
    np_image = ds.pixel_array
    sopclass = ds['0008', '0016'].value
    if sopclass != '1.2.840.10008.5.1.4.1.1.6.1':
        print('Image not a 2d image: {}'.format(file_str))

    photometric_interpretation = ds['0028','0004'].value
    if photometric_interpretation in ["YBR_FULL_422", "RGB"]:
        # Grab the Y or R channel of the image.
        # NOTE: for RGB image, it is assumed that the labels are in the "red channel", so grabbing R channel would suffice
        np_image =  np_image[:, :, 0]
        ds['0028', '0004'].value = 'MONOCRHOME2'
        ds['0028', '0002'].value = 1

    ds.decompress()
    sim = sitk.GetImageFromArray(np_image[:,:])
    sim.SetSpacing(ds.PixelSpacing)
    sitk.WriteImage(sim, target_file_name)

def process_anatomy(anatomy_name_dir):
    data_dir = Path(anatomy_name_dir[1])
    anatomy_name = anatomy_name_dir[0]
    csv_file_name = data_dir / (anatomy_name + '.csv')

    # Read the csv
    original_iamges = []
    try:
        with open(csv_file_name, 'r') as f:
            csv_reader = csv.DictReader(f)
            original_iamges = [ row['original_path'] for row in csv_reader if row['pair_folder'] == 'Pair1' and float(row['meas']) > 0 ]
    except OSError:
        print('ERROR Reading the csv file {}'.format(csv_file_name))

    # Create directory by the name anatomy
    anatomy_folder = data_dir / anatomy_name
    utils.checkDir(anatomy_folder, False)
    # Copy the selected files to the anatomy_img_pair1 folder
    for image in original_iamges:
        full_path = Path('/work/hinashah/data/')/image
        target_path = anatomy_folder / full_path.name
        #if not target_path.exists():
        #    shutil.copyfile(str(full_path), str(target_path))
        target_path_nrrd = str(target_path).replace('dcm', 'nrrd')
        if not Path(target_path_nrrd).exists():
            dcm2nrrd(full_path, target_path_nrrd)
    
    print('Done copying images for anatomy {}, copied {} images'.format(anatomy_name, len(original_iamges)))

    # Pass to runsegmentationOnFolder code
    print('Calling Segmentation/fitting for anatomy {}'.format(anatomy_name))
    run_segmentation_measurement.runSegmentationOnFolder(str(anatomy_folder))
    print('Done with anatomy {}'.format(anatomy_name))

def main(args):
    data_dir = args.dir
    anatomy_names = [ 'HC', 'AC', 'FL']
    anatomy_names_dir = [ [an, data_dir] for an in anatomy_names]
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(process_anatomy, [anatomy_names_dir_in for anatomy_names_dir_in in anatomy_names_dir])
    pool.close()

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with study folders that have prediction.csv files generated', required=True)
    args = parser.parse_args()

    main(args)