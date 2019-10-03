
import itk
import argparse
import os
import glob
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

def main(args):

	InputType = itk.Image[itk.RGBPixel[itk.UC], 3]
	filenames = []

	if(args.dir):
		for img in glob.iglob(os.path.join(args.dir, '**/*.jpg'), recursive=True):
			filenames.append(img)

		labelReader = itk.ImageFileReader.New(FileName=args.label)
		labelReader.Update()
		imglabel = labelReader.GetOutput()

		imglabel_np = itk.GetArrayViewFromImage(imglabel)
		imglabel_np = imglabel_np.flatten()

		imglabel_np = imglabel_np.flatten()
		imglabel_np_indices = np.nonzero(imglabel_np)

		imgtypes = []
		
		for img in filenames:
			
			print("Reading:", img)
			reader = itk.ImageFileReader[InputType].New(FileName=img)
			reader.Update()
			img = reader.GetOutput()

			lumfilter = itk.RGBToLuminanceImageFilter.New(Input=img)
			lumfilter.Update()
			lumimg = lumfilter.GetOutput()

			lumimg_np = itk.GetArrayViewFromImage(lumimg)
			lumimg_np = lumimg_np.flatten()

			lumimg_np_val = np.take(lumimg_np, imglabel_np_indices)

			imgtypes.append(lumimg_np_val.flatten())

		imgtypes = np.array(imgtypes)

		if(args.out_pickle):

			outobj = {}
			outobj["filenames"] = filenames
			outobj["imgtypes"] = imgtypes

			with open(args.out_pickle, 'wb') as f:
				pickle.dump(outobj, f, pickle.HIGHEST_PROTOCOL)

	elif(args.pickle):
		with open(args.pickle,'rb') as f:
			inobj = pickle.load(f)
			filenames = inobj["filenames"]
			imgtypes = inobj["imgtypes"]


	print("Img types shape", imgtypes.shape)
	print("PCA...")
	pca = PCA(n_components=10)
	imgtypes_pca = pca.fit_transform(imgtypes)

	print("PCA shape", imgtypes_pca.shape)
	print("K means n_clusters:", args.n_clusters, "...")
	km = KMeans(n_clusters=args.n_clusters)
	labels = km.fit_predict(imgtypes_pca)
	# af = AffinityPropagation().fit(imgtypes_pca)
	# cluster_centers_indices = af.cluster_centers_indices_
	# labels = af.labels_


	for i, img in enumerate(filenames):
		label = labels[i]

		outdir = os.path.join(args.out, str(label))
		if not os.path.exists(outdir):
			os.makedirs(outdir)

		outfile = os.path.join(outdir, os.path.basename(img))

		print("Creating soft link", outfile)

		if not os.path.exists(outfile):
			os.symlink(img, outfile)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--dir', type=str, help='Directory with jpg images')
	group.add_argument('--pickle', type=str, help='Pickle file')
	parser.add_argument('--label', type=str, help='Input label image, region used for the clustering alg. across all samples.', required=True)
	parser.add_argument('--n_clusters', type=int, help="Number of clusters for kmeans", default=10)
	parser.add_argument('--out', type=str, help='Output directory', default="./")
	parser.add_argument('--out_pickle', type=str, help='Output pickle')
	parser.add_argument('--labelValue', type=int, help='Label value number', default=1)

	args = parser.parse_args()

	main(args)