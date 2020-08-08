from pathlib import Path
import csv
import os
from argparse import ArgumentParser
import numpy as np 
import json
import utils
import logging
import shutil

def main(args):
    if args.test_dir:
        utils.setupLogFile(Path(args.test_dir), args.debug)
        utils.checkDir(Path(args.test_dir)/'misclassified_imgs')

    gt_dir = Path(args.gt_dir)
    if not gt_dir.exists():
        print('The ground truth data directory does not exist, returning')
        return

    gt_tags = ['ac', 'head', 'fl', 'crl']
    pred_tags = ['AC', 'BPD', 'FL', 'CRL', 'R']
    #pred_tags = ['fetus', 'femur', 'head', 'abdomen']
    gt_cases = {}
    csv_predictions = {}
    counts = [0]*len(gt_tags)
    cnt=5000
    try:
        with open(args.pred_csv, 'r') as f:
            csv_reader = csv.DictReader(f)
            for line in csv_reader:
                if args.debug:
                    if cnt ==0:
                        break
                    cnt-=1
                # Get the predicted anatomy
                anatomy = line['class']
                logging.debug('**** Anatomy: {}'.format(anatomy))
                s = line['prediction']
                logging.debug('**** Predictions: {}'.format(s))
                all_pred_conf = [float(t) for t in s.strip('[]').split(',')]
                an_pred_conf = all_pred_conf[  pred_tags.index(anatomy) ]
                logging.debug('***** Confidence for class: {}'.format(an_pred_conf))
                if an_pred_conf > args.conf_thres:
                    norm_path = os.path.normpath(line['img'])
                    logging.debug('**** Image path: {}'.format(norm_path))
                    csv_predictions[norm_path] = {}
                    csv_predictions[norm_path]['pred_an'] = anatomy
                    csv_predictions[norm_path]['pred_conf'] = an_pred_conf

                    if args.pred_dir:
                        pred_case_dir = Path(os.path.normpath((Path(norm_path)).parent))
                        rel_path = pred_case_dir.relative_to(args.pred_dir)
                        case_dir = Path(args.gt_dir)/rel_path
                    else:
                        case_dir = os.path.normpath((Path(norm_path)).parent)

                    if case_dir not in gt_cases:
                        gt_cases[case_dir] = {}
                        gt_cases[case_dir]['frames_paths'] = []
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
                                gt_cases[case_dir]['frames_paths'].append(os.path.normpath( frame_p ))
                                gt_cases[case_dir]['frames'].append(Path(frame_p).stem)
                                gt_cases[case_dir]['gt_an'].append(gt_anatomy)
                                counts[ gt_tags.index(gt_anatomy)  ] +=1
                            logging.debug('////// GT Data setup: {}'.format(case_dir))
                            logging.debug('////// GT FRAMES: {}'.format(gt_cases[case_dir]['frames']))
                            logging.debug('////// GT Anatomy: {}'.format(gt_cases[case_dir]['gt_an']))
                    to_find = Path(norm_path).stem
                    frame_ind = (gt_cases[case_dir]['frames']).index(to_find)
                    logging.debug('TO Find: {}, Frame IND: {}'.format(to_find, frame_ind))
                    if frame_ind < 0:
                        print('Path not found in ground truth csv: gt:{}, pred: {}'.format(gt_csv, norm_path))
                        continue
                    csv_predictions[norm_path]['gt_an'] = gt_cases[case_dir]['gt_an'][frame_ind]
                    logging.debug('Classification tag: {}, GT Tag: {}'.format(csv_predictions[norm_path]['pred_an'], csv_predictions[norm_path]['gt_an']))
    except Exception as e:
        print('Error processing: {}'.format(e))            
    
    print('Counts on the GT images: ')
    print('TAGS:    {}'.format(gt_tags))
    print('Counts:  {}'.format(counts))
    try:
        j_f = json.dumps(csv_predictions, indent=4)
        f = open(str(Path(Path(args.pred_csv).parent)/'gt_pred.json'), "w")
        f.write(j_f)
        f.close()
    except Exception as e:
        print('Error in dumping the json file')
    
    counts = [0.]*len(gt_tags)
    pred_counts = [0.]*len(pred_tags)
    confusion_matrix = np.zeros(shape=(len(gt_tags), len(pred_tags)))
    for pred in csv_predictions:
        d = csv_predictions[pred]
        gt_ind = gt_tags.index( d['gt_an'])
        counts[gt_ind] += 1.0
        pred_ind = pred_tags.index( d['pred_an'])
        pred_counts[pred_ind] += 1.0
        confusion_matrix[gt_ind, pred_ind]+=1
        if (gt_ind != pred_ind) and args.debug and args.test_dir:
            f_name = Path(pred).stem
            suff = Path(pred).suffix
            new_name = f_name + '-' + d['pred_an'] + '-' + d['gt_an'] + suff
            shutil.copyfile(pred, Path(args.test_dir)/Path('misclassified_imgs')/new_name)

    print('Counts on the GT tags after applying threshold {}'.format(args.conf_thres))
    print('TAGS:    {}'.format(gt_tags))
    print('Counts:  {}'.format(counts))


    print('Counts on the Pred tags after applying threshold {}'.format(args.conf_thres))
    print('TAGS:    {}'.format(pred_tags))
    print('Counts:  {}'.format(pred_counts))
    
    # Create the confusion matrix
    print('Raw count confusion matrix')
    print(confusion_matrix)
    for i in range(  confusion_matrix.shape[0] ):
        confusion_matrix[i] = (confusion_matrix[i]/counts[i])*100.0

    print('Percentage confusion matrix:')
    print(confusion_matrix) 

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--gt_dir', type=str, help='Path for the directory with ground truth data', required=True)
    parser.add_argument('--pred_dir', type=str, help='Path to the directory where images that'
                                                    'were input to the classification live (ex resampled images.'
                                                    'Path to these images is saved in the pred_csv file')
    parser.add_argument('--pred_csv', type=str, help='CSV with predicted output', required=True)
    parser.add_argument('--debug', type=bool, help='Enable debug mode, just processes first 15 annotations', default=False)
    parser.add_argument('--conf_thres', type=float, help='Threshold for the predicted class confidence', default=0.95)
    parser.add_argument('--test_dir', type=str, help='Test DIR for outputs')
    args = parser.parse_args()

    main(args)