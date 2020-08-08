from pathlib import Path
from utils import checkDir
from csv import DictWriter
from argparse import ArgumentParser
import json

def read_as_fcs(content):
    splits = content.split(bytes("\x0c", encoding='utf8'))
    k = splits[1::2]
    v = splits[2::2]
    c = [ [k_i.decode('utf-8'), v_i.decode('utf-8')] for (k_i, v_i) in zip(k,v)]
    with open('/Users/hinashah/UNCFAMLI/Data/TREG/Cytoflex_KeyVals.json', 'w') as f:
        json.dump(c, f, indent=4)
    ind = splits.index(b'TBNM')
    uid = splits[ind+1].decode('utf-8')
    ind = splits.index(b'$DATE')
    date = splits[ind+1].decode('utf-8')
    ind = splits.index(b'CH3LBL')
    protocol = splits[ind+1].decode('utf-8')
    return uid, date, protocol

def read_as_lmd(content):
    splits = content.split(bytes("\\", encoding='utf8'))
    k = splits[1::2]
    v = splits[2::2]
    c = [ [k_i.decode('utf-8'), v_i.decode('utf-8')] for (k_i, v_i) in zip(k,v)]
    with open('/Users/hinashah/UNCFAMLI/Data/TREG/FC500_KeyVals.json', 'w') as f:
        json.dump(c, f, indent=4)
    ind = splits.index(b'@SAMPLEID1')
    uid = splits[ind+1].decode('utf-8')
    ind = splits.index(b'$DATE')
    date = splits[ind+1].decode('utf-8')
    return uid, date

def process_list(file_list, out_file_path):
    batch_size = 100
    try:
        file_list_len = len(file_list)
        print('Found {} Files'.format(file_list_len))

        with open(out_file_path, 'a', newline='') as f:
            fieldnames = ['filename', 'patient_id', 'study_date', 'input_type', 'protocol']
            csv_f = DictWriter(f, fieldnames)
            
            cnt=0
            for in_file in file_list:
                cnt +=1
                if cnt%batch_size == 0:
                    print('{}/{}'.format(cnt, file_list_len), end='  ')
                
                if 'FMO' in in_file.name.upper():
                    continue
                
                try:
                    with open(in_file, 'rb') as f:
                        content = f.readlines() 
                        if 'lmd' in in_file.suffix.lower():
                            uid, date = read_as_lmd(content[0])
                            file_type = 'FC500'
                            if 'NEW' in uid.upper():
                                uid = uid.upper().replace('NEW', '').strip()
                            protocol = ''
                        elif 'fcs' in in_file.suffix.lower():
                            uid, date, protocol = read_as_fcs(content[0])
                            file_type = 'CYTOFLEX'
                        else: 
                            raise IOError('Suffix {} not supported, passing'.format(in_file.suffix))
                        
                except Exception as e:
                    print('Error reading file {}, \n error: {}'.format(in_file, e))
                    continue

                row = {'filename':in_file.name, 'patient_id':uid, 'study_date':date, 'input_type': file_type, 'protocol': protocol}
                csv_f.writerow(row)
            print('\n')
    except Exception as e:
        print('Error writing the output csv file. \n error: {}'.format(e))

def main(args):
    data_dir = Path(args.data_dir)
    checkDir(data_dir, False)
    out_file = Path(args.csv_file)
        
    try:
        with open(out_file, 'w', newline='') as f:
            fieldnames = ['filename', 'patient_id', 'study_date', 'input_type', 'protocol']
            csv_f = DictWriter(f, fieldnames)
            csv_f.writeheader()
    except Exception as e:
        print('Error writing the output csv file. \n error: {}'.format(e))

    print('Processing LMDs')
    lmd_files = list(data_dir.glob('**/*.LMD'))
    process_list(lmd_files, out_file)
    
    print('Processing FCS')
    fcs_files = list(data_dir.glob('**/*.fcs'))
    process_list(fcs_files, out_file)
    
    print('----DONE----')


if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--csv_file', type=str, required=True, help='Path to an output.csv')
    parser.add_argument('--data_dir', type=str, help='Input directory location.', required=True)
    args = parser.parse_args()

    main(args)