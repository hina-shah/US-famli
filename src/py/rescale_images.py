from pathlib import Path 
from argparse import ArgumentParser
import os
import json
import SimpleITK as sitk
import numpy as np
import utils
import pydicom
import csv
import gc

def rescale_image(target_shape, image_list, out_image_path):
    for image_path in image_list:
        ext = image_path.suffix
        if ext == '.nrrd':
            im = sitk.ReadImage(str(image_path))
            dimension = im.GetDimension()
            imsize = im.GetSize()
            imspacing = im.GetSpacing()
            if len(imsize) == 3:
                print('Found 3d image: size - {}, filepath: {}'.format(imsize, image_path))
                nparray = sitk.GetArrayFromImage(im)
                nparray = np.squeeze(nparray)
                im = sitk.GetImageFromArray(nparray)
                im.SetSpacing(imspacing[0:2])
        if ext == '.dcm':
            ds = pydicom.read_file(str(image_path))
            # Get a grayscale image
            if ds['0008', '0016'].value == '1.2.840.10008.5.1.4.1.1.6.1':
                photometric_interpretation = ds['0028','0004'].value
                if photometric_interpretation in ["YBR_FULL_422", "RGB"]:
                    np_image = ds.pixel_array[:,:,0]
                elif photometric_interpretation == "MONOCHROME2":
                    np_image = np.squeeze(ds.pixel_array)
                else: 
                    print('Unsupported photometric interpretation for file: {}'.format(image_path))
                    return
            else:
                print('The type of the image not expected here, filepath: {}'.format(image_path))
                return
            im = sitk.GetImageFromArray(np_image)
            im.SetSpacing(ds.PixelSpacing)
            dimension = im.GetDimension()
            image_path = Path( str(image_path).replace('dcm', 'nrrd'))
        new_size = [target_shape]*dimension
        reference_image = sitk.Image(new_size, im.GetPixelIDValue())
        reference_image.SetOrigin(im.GetOrigin())
        reference_image.SetDirection(im.GetDirection())
        reference_image.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(new_size, im.GetSize(), im.GetSpacing())])
        # Resample after Gaussian smoothing.
        imres_smooth = sitk.Resample(sitk.SmoothingRecursiveGaussian(im, 1.0), reference_image)
        sitk.WriteImage(imres_smooth, str(out_image_path/image_path.name))

def main(args):
    json_file = args.json_file
    out_dir = Path(args.out_dir)
    study_list_dict = []
    out_study_list_dict = []
    
    utils.checkDir(out_dir)

    if not out_dir.exists():
        print('The output directory {} does not exist'.format(out_dir))
        return

    try:
        with open(json_file) as jf:
            study_list_dict = json.load(jf)
    except Exception as e:
        print('Error with loading the json file')
        
    print('Found {} number of studies in the json file'.format(len(study_list_dict)))
    num_images = [ int(study['num_cine_images']) for study in study_list_dict]    
    num_bins = 40
    n, bins = np.histogram(np.array(num_images), bins= num_bins)
    print('Histogram of number of cines per study: \n N : {} \n Bin centers: {}'.format(n, bins))

    cnt = 1
    for study in study_list_dict:
        in_study_path = Path(study['study_folder'])
        study_name = in_study_path.name
        cine_class_images_path = in_study_path / 'cine_class_images'
        if not in_study_path.is_dir():
            print('Study folder {} does not exist'.format(in_study_path))
            continue 
        if not cine_class_images_path.is_dir():
            print('Cine classified image dump does not exist in {}'.format(in_study_path))
            continue
        if not (in_study_path / 'head_2d_image.dcm').exists(): 
            print('The 2d image does not exist for study {}'.format(in_study_path))
            continue

        image_list = os.listdir((in_study_path / 'cine_class_images'))
        if len(image_list) != int(study['num_cine_images']):
            print('WARNING: the number of images is not the same as recorded for study: {}'.format(in_study_path))
        images = [cine_class_images_path / image_name for image_name in image_list]
        if( (out_dir / study_name).exists() ):
            print('WARNING: The output study folder: {} already exists'.format(out_dir/study_name))
        out_study_folder = out_dir / study_name / 'cine_class_images'
        out_study_folder.mkdir(parents=True)
        rescale_image(args.target_shape, images, out_study_folder)

        images = [in_study_path / 'head_2d_image.dcm']
        out_study_folder = out_dir/study_name
        rescale_image(args.target_shape, images, out_study_folder)
        
        out_study_dict = {}
        out_study_dict['cine_class_images'] = str(out_dir / study_name / 'cine_class_images')
        out_study_dict['image_2d'] = str(out_dir/study_name/'head_2d_image.nrrd')
        out_study_dict['num_frames'] = study['num_cine_images']
        out_study_list_dict.append(out_study_dict)

        endstr = "\n" if cnt%50 == 0 else " "
        print(".",end=endstr)
        if cnt%50 == 0:
            gc.collect()
        cnt +=1
    try:
        with open( str(out_dir/'dataset_c1_hc_summary.csv'), 'w') as f:
            field_names = ['cine_class_images', 'num_frames', 'image_2d']
            csvwriter = csv.DictWriter(f, field_names)
            csvwriter.writeheader()
            csvwriter.writerows(out_study_list_dict)
    except Exception as e:
        print('ERROR writing the csv file! {}'.format(e))


if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--json_file', type=str, required=True, help='Path to a json file that has a list of the study folders (output of script_populate_classified_cines.py)')
    parser.add_argument('--out_dir', type=str, help='Output directory location.', required=True)
    parser.add_argument('--target_shape', type=int, help = 'Rescaled size of the images (eg. 64 would create 64x64 images)', required=True)
    args = parser.parse_args()

    main(args)
