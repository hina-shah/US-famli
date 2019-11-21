import pandas as pd
import numpy as np
import logging
from .DbBase import DbBase

# This class is for processing sas datasets (or derivatives of)
# that have been generated from dicom sr. 
class SrDb(DbBase):
    
    def __init__(self, path):
        super().__init__(path)
        self._gatagnames = [b'Gestational Age by EDD', b'Gestational Age by LMP']
        self._reqd_tags = ['tagtitle', 'study_id', 'pid', 'scandate', 'tagcontent', 'numeric']

    def checkNeededVariables(self):
        if not super().isValid():
            logging.error('Database not valid, returning')
            return False
    
        varnames_list = self._db.columns.to_list()
        tags_exist = [l in varnames_list for l in self._reqd_tags]
        if tags_exist.count(False) > 0:
            missing_tags = self._reqd_tags[tags_exist == False]
            logging.error('Database is missing required tags: {}'.format(missing_tags))
            return False
        
        return True

    def getGAForStudyID(self, studyID):
        if not super().isValid():
           logging.error('Database not valid, returning')
           return None, None

        subset = self._db[ ['numeric', 'tagtitle' ]][ (self._db['tagtitle'].isin(self._gatagnames)) & 
                                      (self._db['study_id'] == bytes(studyID, 'utf-8')) ]
        if subset.empty:
            return None, None
        else:
            age = subset.iat[0,0]
            gatype = subset.iat[0,1]
            return age, gatype
