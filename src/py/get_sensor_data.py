from pathlib import Path
import os
from argparse import ArgumentParser
import sys
from multiprocessing import Process, Queue, cpu_count
import time
import utils
import json
import subprocess
import shutil
import logging
import csv
from datetime import datetime, timedelta

def main(args):
    take_csv = Path(args.take_csv)
    end_time_txt = Path(args.end_time_txt)
    path_to_study = Path(args.path_to_study)
    all_rows=[]
    with open(take_csv, 'r') as f:
        first_line = f.readline()
        print(first_line)
        
        parts = first_line.split(',')
        ind =  parts.index('Capture Start Time')
        start_time = datetime.strptime(parts[ind+1], '%Y-%m-%d %I.%M.%S.%f %p')
        print(start_time)
        ind = parts.index('Capture Frame Rate')
        capture_time = timedelta(seconds=1./float(parts[ind+1]))
        print(capture_time)

        #skip next 5 lines
        for i in range(6):
            f.readline()
        t = csv.reader(f)
        fieldnames = ['Frame', 'TimeS', 'RotX', 'RotY', 'RotZ', 'PosX', 'PosY', 'PosZ']
        print(fieldnames)
        for row in t:
            row = dict(zip(fieldnames, row))
            d = timedelta(seconds=float(row['TimeS']))
            row['timestamp'] = start_time + d
            all_rows.append(row)
    
    print(len(all_rows))
    cine_end_times = []
    with open(end_time_txt, 'r') as f:
        study_name = f.readline()
        for row in f.readlines():
            if len(row) > 1:
                ts = datetime.strptime(row, '%Y/%m/%d %H:%M:%S:%f \n')
                print(ts)
                cine_end_times.append(ts)
    print(len(cine_end_times))
    
    with open('/Users/hinajoshi/Documents/Work/data/test.txt', 'w') as f:
        f.writelines( [str(ts['timestamp'])+'\n' for ts in all_rows])

    print(cine_end_times[0])
    end_ind = -1
    for i,row in enumerate(all_rows):
        if (row['timestamp'] <= cine_end_times[0] and cine_end_times[0] < row['timestamp'] + capture_time):
            end_ind = i
            break
    
    nframes = 321
    framerate = 31
    cine_rows = []
    if end_ind > 0:
        print(end_ind)
        print(all_rows[end_ind])
        for i in range(nframes):
            frame_ind = end_ind - i*4
            cine_rows.append(all_rows[frame_ind])
        print(len(cine_rows))
        #Output the csv
        cine_rows.reverse()
        csv_rows = []
        for ind, row in enumerate(cine_rows):
            print(ind)
            print(row['timestamp'])
            oprow={}
            oprow['ImageFileName'] = 'frame_{}.nrrd'.format(ind)
            oprow['RotX'] = row['RotX']
            oprow['RotY'] = row['RotY']
            oprow['RotZ'] = row['RotZ']
            oprow['PosX'] = row['PosX']
            oprow['PosY'] = row['PosY']
            oprow['PosZ'] = row['PosZ']
            csv_rows.append(oprow)
        with open(path_to_study/'positions.csv', 'w') as f:
            fieldnames = csv_rows[0].keys()
            writer = csv.DictWriter(f, fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--take_csv', type=str, help='Path to the csv file with the frame information', required=True)
    parser.add_argument('--end_time_txt', type=str, help='Path to the txt file with end times for the study', required=True)
    parser.add_argument('--path_to_study', type=str, help='Path to the study')
    args = parser.parse_args()

    main(args)