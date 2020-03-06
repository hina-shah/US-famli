from argparse import ArgumentParser
from pathlib import Path
import SimpleITK as sitk
import csv
import shutil
import numpy as np
import pydicom

def main(args):
    csv_file = Path(args.csv_file)
    tag_list = (args.tag_list).split()
    som_home_dir_path = Path(args.som_home_dir)
    
    try:
        tag_lines = []
        with open(csv_file, 'r') as f:
            csv_reader = csv.DictReader(f)
            tag_lines = [line for line in csv_reader if line['tag'] in tag_list]
        print('Found: {} tag lines'.format(len(tag_lines)))
        for tag_line in tag_lines:
            tag_file_path = Path( som_home_dir_path / tag_line['File'])
            if tag_file_path.exists():
                ds = pydicom.read_file(str(tag_file_path))
                if ds is None:
                    print('File: {} Missing DICOM metadata'.format(str(tag_file_path)))
                photometric_interpretation = ds['0028','0004'].value
                print('Photometric interpretaion: {}'.format(photometric_interpretation))
                #print('Shape of pixel data: {}'.format(ds.pixel_array.shape))
                np_image = ds.pixel_array
                if photometric_interpretation in ["YBR_FULL_422", "RGB"]:
                    np_image = np_image[:,:,:,0]
                
                file_out = str(Path(args.out_dir) / (tag_line['tag'] + '.dcm'))
                out_spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), 1.]
                sim = sitk.GetImageFromArray(np_image)
                sim.SetSpacing(out_spacing)
                sitk.WriteImage(sim, file_out)

    except Exception as e:
        print('Error with processing csv file: {} \n {}'.format(csv_file, e))
    print('____DONE_____')


if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--csv_file', type=str, required=True, help='Path to an info.csv file that has the OCR tags')
    parser.add_argument('--out_dir', type=str, help='Output directory location.', required=True)
    parser.add_argument('--som_home_dir', type=str, help = 'Mount location for the SOM server.' 
                                'Basically parent location for the famli folder, and original ultrasound images are stored here', required=True)
    parser.add_argument('--tag_list', default=None, help='List of tags to use from the info file')
    args = parser.parse_args()

    main(args)
