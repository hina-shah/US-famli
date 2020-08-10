from pathlib import Path
import SimpleITK as sitk 
from argparse import ArgumentParser
import numpy as np
import itk
import os

# THIS script takes as input output from run_segmentation_measurement.py 
# Basically given two folders A and B which would have images of the same names (ex HC, HC_seg in a study)
# This will iso-resample the images of A to size of images of B, 
# and then will create 3d images from frames of both the folders as A_3d.nrrd, and B_3d.nrrds

parser = ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--dir_seg', type=str)
parser.add_argument('--out_dir', type=str)
args = parser.parse_args()

img_folder = Path(args.dir)
img_seg_folder = Path(args.dir_seg)
out_folder = Path(args.out_dir)
print('Out Folder: {}'.format(out_folder))

img_list = list(img_folder.glob('**/*.nrrd'))

first_im  = sitk.ReadImage( str(img_folder/img_list[0].name))
first_im_seg = sitk.ReadImage(str(img_seg_folder/img_list[0].name))

target_size_2d = first_im_seg.GetSize()
orig_spacing = first_im.GetSpacing()
orig_size = first_im.GetSize()
target_spacing = [ os*osz/tsz for os, osz, tsz in zip(orig_spacing, orig_size, target_size_2d)  ]
max_target_spacing = max(target_spacing)
target_spacing_2d = first_im_seg.GetSpacing() # [max_target_spacing, max_target_spacing]
print('Target spacing: {}'.format(target_spacing))
print('Target size: {}'.format(target_size_2d))

print('Resampling first')
for img_name in img_list:
    print(img_name)
    sim = sitk.ReadImage(str(img_name))
    sitk_resample = sitk.ResampleImageFilter()
    sitk_resample.SetOutputSpacing(target_spacing_2d)
    sitk_resample.SetOutputDirection(sim.GetDirection())
    sitk_resample.SetSize(target_size_2d)
    sitk_out_img = sitk_resample.Execute(sim)
    print('Writing resampled image: {}'.format(out_folder/img_name.name))
    print(sitk_out_img.GetSize())
    print(sitk_out_img.GetSpacing())
    sitk.WriteImage(sitk_out_img, str(out_folder / img_name.name))

print('Converting to 3d')

for ip_dir in [out_folder, img_seg_folder]:
    InputPixelType = itk.F
    InputImageType = itk.Image[InputPixelType, 2]

    img_read = itk.ImageFileReader[InputImageType].New(FileName=str(ip_dir / img_list[0].name))
    img_read.Update()
    img = img_read.GetOutput()

    PixelType = itk.template(img)[1][0]
    OutputImageType = itk.Image[PixelType, 2]

    PixelType = itk.template(img)[1][0]
    OutputImageType = itk.Image[PixelType, 3]
    InputImageType = itk.Image[PixelType, 2]

    tileFilter = itk.TileImageFilter[InputImageType, OutputImageType].New()

    layout = [1, 1, 0]
    tileFilter.SetLayout(layout)

    imgs_arr_description = []

    for i, img_name in enumerate(img_list):
        filename = str(ip_dir/img_name.name)
        print("Reading:", filename)
        img_read = itk.ImageFileReader[InputImageType].New(FileName=filename)
        img_read.Update()
        img = img_read.GetOutput()

        tileFilter.SetInput(i, img)

        img_obj_description = {}
        img_obj_description["image"] = {}
        img_obj_description["image"]["region"] = {}
        img_obj_description["image"]["region"]["size"] = np.array(img.GetLargestPossibleRegion().GetSize()).tolist()
        img_obj_description["image"]["region"]["index"] = np.array(img.GetLargestPossibleRegion().GetIndex()).tolist()
        img_obj_description["image"]["spacing"] = np.array(img.GetSpacing()).tolist()
        img_obj_description["image"]["origin"] = np.array(img.GetOrigin()).tolist() 
        img_obj_description["image"]["direction"] = itk.GetArrayFromVnlMatrix(img.GetDirection().GetVnlMatrix().as_matrix()).tolist()
        img_obj_description["img_filename"] = os.path.basename(filename)

        imgs_arr_description.append(img_obj_description)

    defaultValue = 0
    tileFilter.SetDefaultPixelValue(defaultValue)
    tileFilter.Update()

    writer = itk.ImageFileWriter[OutputImageType].New()
    out_im_name = ip_dir.stem + '_3d.nrrd'
    print('Writing 3d image: {}'.format(out_folder/out_im_name))
    writer.SetFileName(str(out_folder/out_im_name))
    writer.SetInput(tileFilter.GetOutput())
    writer.Update()
