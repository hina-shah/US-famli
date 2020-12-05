from pathlib import Path
from sys import platform
import logging
from csvwrap import CSVWrap

class TagInfoFile():
    '''
    This file handles the reading/writing of the
    info csv files in the FAMLI dataset C1 studies
    '''
    file_name = 'tags.csv'

    def __init__(self, study_dir):
        
        # study dir is the output folder where the tag file will be written
        if isinstance(study_dir, str):
            self.study_dir = Path(study_dir)
        elif isinstance(study_dir, Path):
            self.study_dir = study_dir
        else:
            raise TypeError(('Study DIR path object is of the wrong type'))

        self.file_name = TagInfoFile.file_name
        self.tag_info_file = self.study_dir/TagInfoFile.file_name
        self.file_tag_dict = []

    def get_file_path(self):
        return tag_info_file
    
    def check_keys(self, keys_list):
        return 'File' in keys_list and 'type' in keys_list and 'tag' in keys_list

    def read(self):
        try:
            rows, keys = CSVWrap.readCSV(self.tag_info_file )
            if rows is not None and self.check_keys(keys):
                self.file_tag_dict = rows
        except Exception as e:
            logging.error('Error reading the tag file')
            self.file_tag_dict = []

    def write(self):
        try:
            if len(self.file_tag_dict) == 0:
                logging.warning('Nothing to write in taginfo, skipping')
                return
            CSVWrap.writeCSV(self.file_tag_dict, self.tag_info_file)
        except Exception as e:
            logging.error('Error writing the tag file')

    def addTag(self, abs_study_original_path, relative_to_path, file_name, filetype, tag, write=False):
        """
        abs_study_original_path: Absolute path to the study
        relative_to_path: Path to the server. Study's relative path to the server will be stored
        file_name: Name of the file in study
        """
        # study dir is the output folder where the tag file will be written
        if isinstance(abs_study_original_path, str):
            abs_study_original_path = Path(abs_study_original_path)
        elif not isinstance(abs_study_original_path, Path):
            raise TypeError('Input absolute study original path has to be either string or a Path')

        relative_study_path = abs_study_original_path.relative_to(relative_to_path)

        row = {}
        row['File'] = relative_study_path / file_name
        row['type'] = filetype
        row['tag'] = tag

        self.file_tag_dict.append(row)
        if write:            
            CSVWrap.writeCSV([row], self.tag_info_file, append=True)

    def getFileNamesWithTag(self, tag):
        file_names = [ file_row['File']  for file_row in self.file_tag_dict if file_row['tag'] == tag]
        return file_names

    def getFileNamesWithType(self, filetype):
        file_names = [ file_row['File']  for file_row in self.file_tag_dict if file_row['type'] == filetype]
        return file_names

    def getTagForFileName(self, file_name):
        for file_row in self.file_tag_dict:
            if file_row['File'].name == file_name:
                return file_row['tag']

    def getStudyDir(self):
        return self.study_dir

    def setStudyDIr(self, study_dir):
        if isinstance(study_dir, str):
            self.study_dir = Path(study_dir)
        elif isinstance(study_dir, Path):
            self.study_dir = study_dir
        else:
            raise TypeError(('Study DIR path object is of the wrong type'))
    
    def getNumFiles(self):
        return len(self.file_tag_dict)