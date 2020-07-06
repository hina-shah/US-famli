from argparse import ArgumentParser
from pathlib import Path
import SimpleITK as sitk
import csv
import shutil
import numpy as np
import pydicom
import utils
from multiprocessing import Process, Queue, cpu_count, Pool, Lock

def processOneCsv( argslist ):
    msg = ''
    try:
        if len(argslist) !=4:
            raise Exception('Wrong number of arguments')
        
        csv_file = argslist[0]
        tag_list = argslist[1]
        som_home_dir_path = argslist[2]
        out_dir = argslist[3]

        tag_lines = []
        with open(csv_file, 'r') as f:
            csv_reader = csv.DictReader(f)
            tag_lines = [line for line in csv_reader if line['tag'] in tag_list]
        msg = 'Found: {} tag lines'.format(len(tag_lines)) + '\n'
        for tag_line in tag_lines:
            tag_file_path = Path( som_home_dir_path / tag_line['File'])
            tag_file_path.resolve()
            if tag_file_path.exists():
                file_out_path = out_dir / (tag_line['tag'] + '.dcm')
                file_out = str(file_out_path)
                if file_out_path.exists():
                    msg = 'Skipping {}, already exists'.format(file_out)
                    continue

                ds = pydicom.read_file(str(tag_file_path))
                if ds is None:
                    raise Exception('File: {} Missing DICOM metadata'.format(str(tag_file_path)))

                photometric_interpretation = ds['0028','0004'].value
                #print('Shape of pixel data: {}'.format(ds.pixel_array.shape))
                np_image = ds.pixel_array
                if photometric_interpretation in ["YBR_FULL_422", "RGB"]:
                    np_image = np_image[:,:,:,0]

                out_spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), 1.]
                sim = sitk.GetImageFromArray(np_image)
                sim.SetSpacing(out_spacing)
                sitk.WriteImage(sim, file_out)
            else:
                raise Exception('Could not find path: {}'.format(tag_file_path))

    except Exception as e:
        msg = 'Error with processing csv file: {} \n {}'.format(csv_file, e)
    
    return msg

def map_queue_to_pool(case_queue):
    """
    Map the the queues to a thread pool for parallel processing to dump frames
    """
    if case_queue is None or len(case_queue) == 0:
        return
    pool = Pool(len(case_queue))
    results = pool.map(processOneCsv, [case for case in case_queue])
    pool.close()
    pool.join()
    for i,msg in enumerate(results):
        if len(msg) > 0:
            print('Pool thread: {}, MSG: {}'.format(i, msg))

def convert_to_gray_proc(queue):
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

def main(args):
    som_home_dir_path = Path(args.som_home_dir)
    if not args.tag_list:
        tag_list_file = 'us_cine_tags.txt'
        try:
            with open(Path(__file__).parent / tag_list_file) as f:
                tag_list = f.read().splitlines()
        except:
            print('ERROR READING THE TAG FILE')
            tag_list = ['M', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'R15', 'R45', 'R0', 'RO', 'R1',
                        'L15', 'L45', 'L0', 'LO', 'L1', 'M0', 'M1', 'RTA', 'RTB', 'RTC']
    else:
        tag_list = (args.tag_list).split()

    if not args.csv_file and not args.in_dir:
        print('Need either a csv_file or in_dir in arguments. Exiting')
        return
    
    if args.csv_file:
        csv_file = Path(args.csv_file)
        msg = processOneCsv( [csv_file, tag_list, som_home_dir_path, Path(args.out_dir)])
        print(msg)
        return
    
    # Directory mode.
    # Find all info_corrected.csv files in the in_dir
    in_dir = Path(args.in_dir)
    all_csvs = list( in_dir.glob('**/info_corrected.csv') )
    out_dir = Path(args.out_dir)
    utils.checkDir(out_dir, False)

    squeue =  Queue()
    convert_to_gray_p = Process(target=convert_to_gray_proc, args=(squeue,))
    convert_to_gray_p.start()

    for csv_file in all_csvs:
        # Estimate the output directory, and create if it doesn't exist
        study_dir = csv_file.parent
        rel_path = study_dir.relative_to(in_dir)
        study_out = out_dir/rel_path
        utils.checkDir(study_out, False)
        squeue.put( [csv_file, tag_list, som_home_dir_path, study_out])
    squeue.put('DONE')
    convert_to_gray_p.join()
    hprint('____DONE_____')

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--in_dir', type=str, help='Path to the directory where info_corrected csv files live, this will'
                                                'This will copy the directory structure in the out_dir and store results there' )
    parser.add_argument('--csv_file', type=str, help='Path to an info.csv file that has the OCR tags')
    parser.add_argument('--out_dir', type=str, help='Output directory location.', required=True)
    parser.add_argument('--som_home_dir', type=str, help = 'Mount location for the SOM server.' 
                                'Basically parent location for the famli folder, and original ultrasound images are stored here', required=True)
    parser.add_argument('--tag_list', default=None, help='List of tags to use from the info file')
    args = parser.parse_args()

    main(args)
