
import itk
import argparse

parser = argparse.ArgumentParser(description='U net segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


in_group = parser.add_mutually_exclusive_group(required=True)
  
in_group.add_argument('--img', type=str, help='Input image for prediction')
in_group.add_argument('--dir', type=str, help='Directory with images for prediction')

parser.add_argument('--resize', nargs="+", type=int, help='Resize images during prediction, useful when doing whole directories with images of diferent sizes. This is needed to set the value of the placeholder. e.x. 1 1500 1500 1', default=None)

parser.add_argument('--out', type=str, help='Output image or directory. If dir flag is used the output image name will be the <Directory set in out flag>/<imgage filename in directory dir>', default="./out.nrrd")

args = parser.parse_args()

if(args.img):
  print('imgage_name', args.img)
  fobj = {}
  fobj["img"] = args.img
  fobj["out"] = args.out
  if args.ow or not os.path.exists(fobj["out"]):
    filenames.append(fobj)
elif(args.dir):
  print('dir', args.dir)
  for img in glob.iglob(os.path.join(args.dir, '**/*'), recursive=True):
    if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".jpg", ".png"]]:
      fobj = {}
      fobj["img"] = img
      fobj["out"] = os.path.join(args.out, img.replace(args.dir, ''))
      if not os.path.exists(os.path.dirname(fobj["out"])):
        os.makedirs(os.path.dirname(fobj["out"]))

      if args.ow or not os.path.exists(fobj["out"]):
        filenames.append(fobj)


resize_shape = args.resize

InputType = itk.Image[itk.F,2]
img_read = itk.ImageFileReader[InputType].New(FileName=filenames[0]["img"])
img_read.Update()
img = img_read.GetOutput()

size = resize_shape
spacing = img.GetSpacing()

centralPixel = itk.Index[dimension]()
centralPixel[0] = int(size[0] / 2)
centralPixel[1] = int(size[1] / 2)
centralPoint = itk.Point[itk.D, dimension]()
centralPoint[0] = centralPixel[0]
centralPoint[1] = centralPixel[1]

scaleTransform = itk.ScaleTransform[itk.D, dimension].New()

parameters = scaleTransform.GetParameters()
parameters[0] = 1
parameters[1] = 1

scaleTransform.SetParameters(parameters)
scaleTransform.SetCenter(centralPoint)

interpolatorType = itk.LinearInterpolateImageFunction[type(img), itk.D]
interpolator = interpolatorType.New()

resamplerType = itk.ResampleImageFilter[type(img), type(img)]
resampleFilter = resamplerType.New()

resampleFilter.SetInput(img)
resampleFilter.SetTransform(scaleTransform)
resampleFilter.SetInterpolator(interpolator)
resampleFilter.SetSize(size)
resampleFilter.SetOutputSpacing(spacing)
resampleFilter.Update()
img = resampleFilter.GetOutput()

print("Writing:", img_obj["out"])
writer = itk.ImageFileWriter.New(FileName=img_obj["out"], Input=img)
writer.Update()