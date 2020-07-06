from argparse import ArgumentParser
from pathlib import Path
import json
import pandas as pd
from multiprocessing import Process, Queue, cpu_count, Pool, Lock
import utils
from dump_images_from_cine import dumpImagesForCine

def dumpFramesForOneCase(case_dict):
    msg = ''
    panda_df = case_dict['df']
    video_file = case_dict['file']
    frame_list = panda_df['frame'].tolist()
    out_dir = case_dict['out']
    try:
        path_to_video = Path(video_file)
        if (path_to_video.suffix).lower() in  ['.dcm', '.mp4']:
            out_path, msg = dumpImagesForCine(path_to_video, Path(out_dir), store_jpg = True, frame_list = frame_list, verbatim=False)
            if out_path is not None:
                frame_paths = [ str(out_path/("frame_{}.jpg".format(l))) for l in panda_df['frame'].tolist() ]
                panda_df['frame_path'] = frame_paths
                panda_df.to_csv(str(out_path/'gt_frames.csv'))
    except Exception as e:
        msg = "Exception during dump frames::=> {}".format(e)
    return msg

def map_queue_to_pool(case_queue):
    """
    Map the the queues to a thread pool for parallel processing to dump frames
    """
    if case_queue is None or len(case_queue) == 0:
        return
    pool = Pool(len(case_queue))
    results = pool.map(dumpFramesForOneCase, [case for case in case_queue])
    pool.close()
    pool.join()
    for i,msg in enumerate(results):
        if len(msg) > 0:
            print('Pool thread: {}, MSG: {}'.format(i, msg))

def frame_dump_proc(queue):
    max_num_processes = cpu_count()
    print('Using {} number of processes'.format(max_num_processes))
    internal_queue = []
    pool_count=0
    while True:
        study = queue.get()
        if study == 'DONE':
            # Process the remaining internal_queue
            print('starting final pool, length {}'.format(len(internal_queue)))
            map_queue_to_pool(internal_queue)
            break
        if len(internal_queue) < max_num_processes:
            internal_queue.append(study)
        
        if len(internal_queue) == max_num_processes:
            print('starting a pool # {} (num of studies: {})'.format(pool_count, pool_count*max_num_processes))
            pool_count +=1
            map_queue_to_pool(internal_queue)
            internal_queue = []

def getFrameList(annotation_list, annotators, n_frames_thres):
    '''
    This function takes a list of annotations, and looks for the 'measurability' 
    tag in the results tag.
    It will pick out the frames for this annotation task, 
        if the number of frames is more than 1
        if the annotator is an approved one
    Inputs: 
    annotation_list: a list of annotations with 'results' and 'measurability' tags. Read from butterfly json
    annotators: list of approved annotators' user names
    n_frames_thres: is a threshold for min number of frames that should have been annotated for 
                    an annotation to be considered

    Returns: A dataframe with one column of frames, and the other of the visible anatomy
    '''
    task_frames_dict = {}
    for a in annotation_list:
        if 'results' not in a or 'measurability' not in a['results']:
            print('Results or measurability not found in this annotation')
            continue
        annotator = a['username']
        if annotator in annotators:
            measures = a['results']['measurability']
            if len(measures) >= n_frames_thres:
                frames = { m[0] for m in measures if m[1][0] in ['measurable', 'visible'] }
                # Find the name of the anatomy
                # Using All measurability/visibility tasks. Not separating them. 
                # If an anatomy is measurable it will be visibile as well
                # Hence, just create task with the name of the anatomy
                task = (a['task'])[:a['task'].find('-')]
                # if there's already a list of frames for this anatomy, add more.
                if task in task_frames_dict:
                    task_frames_dict[task] = task_frames_dict[task].union(frames)
                else:
                    task_frames_dict[task] = frames

    if len(task_frames_dict) > 0:
        frame_list = {}
        frame_list['frame'] = []
        frame_list['anatomy'] = []
        for task in task_frames_dict:           
            frame_list['anatomy'].extend( [task]*len(task_frames_dict[task]))
            frame_list['frame'].extend(list(task_frames_dict[task]))
        case_df = pd.DataFrame.from_dict(frame_list)
        return case_df
    else:
        return None

def main(args):
    json_file = Path(args.json_file)
    out_images_dir = Path(args.out_dir)
    path_to_server = Path(args.server_path)
    utils.checkDir(out_images_dir, False)
    
    if not json_file.exists():
        print('Wrong path for the json file, try again')
        return
    
    files_list = []
    try:
        with open(json_file, 'r') as f:
            annot_dict = json.load(f)
            files_list = annot_dict['files']
    except Exception as e:
        print('Error reading the json file: {}'.format(e))
        return
    
    print("Number of elements in the input JSON: {}".format(len(files_list)))

    # Create another process that accepts a queue of cases to dump frames
    if args.dump_frames:
        squeue =  Queue()
        frame_dump_p = Process(target=frame_dump_proc, args=(squeue,))
        frame_dump_p.start()

    # Look for measured/visible anntoations
    cases_with_ann =  [s for s in files_list if 'annotations' in s] 
    annt_res = 0
    annt_meas = 0
    results_keys = []
    cine_paths = set()
    for case in cases_with_ann:
        annotations = case['annotations']
        with_results = [a for a in annotations if 'results' in a]
        with_meas = [ a for a in annotations if ('results' in a and 'measurability' in a['results']) ]
        keys = [ ]
        for a in with_results:
            keys.extend(list(a['results'].keys()))
    
        if len(with_meas) > 0:
            cine_paths.add(case['file_key'])
            cine_path = path_to_server/case['file_key']
            # Check if this case has been processed
            expected_out = out_images_dir/Path(cine_path.stem)/'gt_frames.csv'
            if expected_out.exists():
                print('Cine {} already processed, skipping'.format(cine_path))
                annt_res+=1
                continue
            # Some annotations are present
            # Call the function to get the final frame list from the annotations.
            df = getFrameList(with_meas, ['enam', 'adiaz', 'arellanes'], args.n_frames_thres)
            if df is not None:
                # Call the frames dump routine - this should be the threaded version.
                case_dict = { 'file':str(cine_path), 'out':str(out_images_dir), 'df':df}
                squeue.put(case_dict)
                annt_res +=1
        if annt_res==15 and args.debug:
            break
    
    squeue.put('DONE')
    frame_dump_p.join()

    print("Number of cases with annotations: {}".format(len(cases_with_ann)))
    print("Number of cases with measurement results in annotations: {}".format(len(cine_paths)))
    print("Number of cases with measurements from approved users and no of frames >= {}: {}".format(args.n_frames_thres, annt_res))
    
    with open(out_images_dir/'cine_list.txt', 'w') as f:
        f.writelines("%s\n" % file_name for file_name in cine_paths)

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--json_file', type=str, help='Path to annotation json file', required=True)
    parser.add_argument('--n_frames_thres', type=int, help='Min number of frames in a task' 
                                                            'Annotations with >= n_frames_thres will be used', \
                                                    default=2 )
    parser.add_argument('--out_dir', type=str, help='Path for frame dump output', required=True)
    parser.add_argument('--server_path', type=str, help='Path to the server space where data exists', required=True)
    parser.add_argument('--debug', type=bool, help='Enable debug mode, just processes first 15 annotations', default=False)
    parser.add_argument('--dump_frames', type=bool, help='Dum frames?', default=True)
    args = parser.parse_args()

    main(args)