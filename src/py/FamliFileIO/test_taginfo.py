import unittest
from pathlib import Path
from taginfo import TagInfoFile

class TagInfoFileTest(unittest.TestCase):
    def test(self):
        study_path = '/Users/hinashah/UNCFAMLI/Data/us_tags_test/2018-09/UNC-0026-1_20180924_093918'

        taginfoinst = TagInfoFile(study_path)
        taginfoinst.addTag('/Users/hinashah/UNCFAMLI/Data/Ultrasound_data/2018-09/UNC-0026-1_20180924_093918', '/Users/hinashah/UNCFAMLI', 'row1.dcm', '2d image', 'ac', write = True)
        taginfoinst.addTag('/Users/hinashah/UNCFAMLI/Data/Ultrasound_data/2018-09/UNC-0026-1_20180924_093918', '/Users/hinashah/UNCFAMLI', 'row2.dcm', 'ge kretz', 'eq', write = True)
        taginfoinst.addTag('/Users/hinashah/UNCFAMLI/Data/Ultrasound_data/2018-09/UNC-0026-1_20180924_093918', '/Users/hinashah/UNCFAMLI', 'row3.dcm', 'cine', 'M', write = True)
        taginfoinst.addTag('/Users/hinashah/UNCFAMLI/Data/Ultrasound_data/2018-09/UNC-0026-1_20180924_093918', '/Users/hinashah/UNCFAMLI', 'row4.dcm', 'cine', 'M', write = True)
        taginfoinst.addTag('/Users/hinashah/UNCFAMLI/Data/Ultrasound_data/2018-09/UNC-0026-1_20180924_093918', '/Users/hinashah/UNCFAMLI', 'row5.dcm', 'SR', 'unknown', write=True)

        self.assertEqual(taginfoinst.getNumFiles(), 5)
        self.assertEqual( len(taginfoinst.getTagForFileName('row3.dcm')), 1)
        self.assertEqual( len(taginfoinst.getFileNamesWithTag('M')), 2)
if __name__ == '__main__':
    unittest.main()