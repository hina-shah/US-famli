from argparse import ArgumentParser
from pathlib import Path
import logging
import os
import csv
import utils
import db

def main(args):
    pass



if __name__=="__main__":
# /Users/hinashah/famli/Groups/Restricted_access_data/Clinical_Data/EPIC/Dataset_B
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory with the EPIC database files')
    parser.add_argument('--out_file', type=str, help='Output database filename')
    parser.add_argument('--debug', action='store_true', help='Add debug info in log')
    args = parser.parse_args()

    main(args)