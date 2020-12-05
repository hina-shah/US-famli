from pathlib import Path
import csv
import logging

class CSVWrap():
    '''
    This class wraps csv reading/writing

    It handles the dict reading, writing and record handling.
    '''

    def __init__(self, filepath, csvrows=None):
        # read filename
        if isinstance(filepath, str):
            self.file_path = Path(filepath)
        elif isinstance(filepath, Path):
            self.file_path = filepath
        else:
            raise TypeError(('CSV input File path object is of the wrong type'))
        
        if csvrows is None:
            self.csvrows = []
            self.keys = []
        else:
            if not isinstance(csvrows, dict):
                logging.warning('Expecting a dict while creating csv rows')
                self.csvrows = []
                self.keys = []
            else:
                self.keys = csvrows[0].keys()
                self.csvrows = csvrows
        self.numrows_written = 0
    
    def append_row(self, rowdict, write=False):
        # Expecting a dict
        if not isinstance(rowdict, dict):
                raise TypeError('Expecting a dict row to append to the csv')
            
        if len(self.csvrows) == 0:
            # initiate the keys from the first row
            self.keys = rowdict.keys()
        else:
            rowkeys = rowdict.keys()
            if rowkeys != self.keys:
                raise TypeError('Do not have the right keys for this file') 
            self.csvrows.append(rowdict)
            if write:
                CSVWrap.writeCSV([rowdict], self.file_path, append=True)

    def read(self):
        try:
            self.csvrows, self.keys = CSVWrap.readCSV(self.file_path)
        except Exception as e:
            logging.error('Error reading the file: {} \n {}'.format(self.file_path, e))
            self.csvrows = []
            self.keys = []

    def get_rows(self):
        return self.csvrows

    def get_numrows(self):
        return len(self.csvrows)

    def get_header(self):
        return self.keys

    def reset_rows(self):
        self.csvrows = []
        self.keys = []

    def write(self):
        try:
            CSVWrap.writeCSV(self.csvrows, self.file_path)
        except Exception as e:
            logging.error('Error while writing the file')

    @staticmethod
    def writeCSV(csvrows, filepath, append=False, newline=''):
        if len(csvrows) == 0:
            logging.warning('No information to export to csv')
            return

        firstrow = csvrows[0]

        if not isinstance(filepath, Path):
            raise TypeError('Expecting a Path to be passed in for writing')

        if not isinstance(firstrow, dict):
            raise TypeError('Expecting a list of row dicts')
        try:
            mode = 'a' if append else 'w'
            if not filepath.exists():
                # Will need the write mode while file generation to crete 
                # the header row
                mode = 'w'
            fieldnames = csvrows[0].keys()
            with open(filepath, mode, newline=newline) as fh:
                csv_writer = csv.DictWriter(fh,fieldnames)
                if mode == 'w':
                    csv_writer.writeheader()
                csv_writer.writerows(csvrows)
        except Exception as e:
            logging.error('File: {}, error writing the rows \n {}'.format(filepath, e))
    
    @staticmethod
    def readCSV(filepath):
        if not isinstance(filepath, Path):
            logging.error('Expecting a Path to be passed in for reading')
            return None, None
        if not filepath.exists() or filepath.suffix != '.csv':
            logging.error('Either the file does not exist, or wrong type of file')
            return None, None

        csvrows = []
        try:
            with open(filepath, 'r') as fh:
                csv_reader = csv.DictReader(fh)
                for line in csv_reader:
                    csvrows.append(line)
        except Exception as e:
            logging.error('File: {}, error writing the rows'.format(filepath))
            return None, None
        
        if len(csvrows) > 0:
            return csvrows, csvrows[0].keys()
        else:
            return None, None
