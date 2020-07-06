from pathlib import Path
import csv
import os
from argparse import ArgumentParser
import numpy as np 
import json

def main(args):
    input_dir = Path(args.frames_dir)
    if not input_dir.exists():
        print('The input directory does not exist, returning')
        return
    gt_tags = ['crl', 'fl', 'head', 'ac']
    pred_tags = ['fetus', 'femur', 'head', 'abdomen']
    gt_cases = {}
    csv_predictions = {}
    counts = [0]*len(gt_tags)
    try:
        with open(args.pred_csv, 'r') as f:
            csv_reader = csv.DictReader(f)
            for line in csv_reader:
                anatomy = line['class']
                if float(line[anatomy]) > args.conf_thres:
                    norm_path = os.path.normpath(line['img'])
                    csv_predictions[norm_path] = {}
                    csv_predictions[norm_path]['pred_an'] = anatomy
                    csv_predictions[norm_path]['pred_conf'] = float(line[anatomy])
                    case_dir = os.path.normpath((Path(norm_path)).parent)
                    if case_dir not in gt_cases:
                        gt_cases[case_dir] = {}
                        gt_cases[case_dir]['frames'] = []
                        gt_cases[case_dir]['gt_an'] = []
                        gt_csv = Path(case_dir)/'gt_frames.csv'
                        if not gt_csv.exists():
                            print('Corresponding ground truth csv does not exist for parent: {}'.format(case_dir))
                            continue
                        with open(gt_csv, 'r') as g:
                            g_reader = csv.DictReader(g)
                            for frame_line in g_reader:
                                gt_anatomy = (frame_line['anatomy']).lower()
                                gt_anatomy = 'head' if gt_anatomy in ['tcd'] else gt_anatomy
                                frame_p = frame_line['frame_path'] #.replace('Annotated_Cine_Frames', 'Annotated_Cine_Frames_Sub')
                                gt_cases[case_dir]['frames'].append(os.path.normpath( frame_p ))
                                gt_cases[case_dir]['gt_an'].append(gt_anatomy)
                                counts[ gt_tags.index(gt_anatomy)  ] +=1
                    frame_ind = (gt_cases[case_dir]['frames']).index(norm_path)
                    if frame_ind < 0:
                        print('Path not found in ground truth csv: gt:{}, pred: {}'.format(gt_csv, norm_path))
                        continue
                    csv_predictions[norm_path]['gt_an'] = gt_cases[case_dir]['gt_an'][frame_ind]
    except Exception as e:
        print('Error processing: {}'.format(e))            
    
    print(gt_tags)
    print('Counts on the GT images: ')
    print(counts)
    try:
        j_f = json.dumps(csv_predictions, indent=4)
        f = open(str(Path(Path(args.pred_csv).parent)/'gt_pred.json'), "w")
        f.write(j_f)
        f.close()
    except Exception as e:
        print('Error in dumping the json file')
    
    counts = [0.]*len(gt_tags)
    confusion_matrix = np.zeros(shape=(len(gt_tags), len(pred_tags)))
    for pred in csv_predictions:
        d = csv_predictions[pred]
        gt_ind = gt_tags.index( d['gt_an'])
        counts[gt_ind] += 1.0
        pred_ind = pred_tags.index( d['pred_an'])
        confusion_matrix[gt_ind, pred_ind]+=1

    print('Counts on the GT tags after applying threshold {}'.format(args.conf_thres))
    print(counts)

    # Create the confusion matrix
    print('Raw count confusion matrix')
    print(confusion_matrix)
    for i in range(  confusion_matrix.shape[0] ):
        confusion_matrix[i] = (confusion_matrix[i]/counts[i])*100.0

    print('Percentage confusion matrix:')
    print(confusion_matrix) 

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--frames_dir', type=str, help='Path for frame dump output', required=True)
    parser.add_argument('--pred_csv', type=str, help='Path to the server space where data exists', required=True)
    parser.add_argument('--debug', type=bool, help='Enable debug mode, just processes first 15 annotations', default=False)
    parser.add_argument('--conf_thres', type=float, help='Threshold for the predicted class confidence', default=0.95)
    args = parser.parse_args()

    main(args)