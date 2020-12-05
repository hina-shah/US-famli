import unittest
from pathlib import Path
from csvwrap import CSVWrap

class CSVWrapTest(unittest.TestCase):
    def test(self):
        csv_path = '/Users/hinashah/UNCFAMLI/Data/us_tags_test/2018-09/UNC-0026-1_20180924_093918/info_corrected.csv'
        # Test reading.
        csvrows, header = CSVWrap.readCSV(Path(csv_path))
        self.assertEqual(len(csvrows), 54)
        self.assertEqual(len(header), 3)

        csvinst = CSVWrap(csv_path)
        csvinst.read()
        self.assertEqual(csvinst.get_numrows(), 54)
        self.assertEqual(len(csvinst.get_header()), 3)

        newrow = {}
        newrow['File'] = 'a'
        newrow['type'] = 't'
        newrow['tag'] = 'ta'
        self.assertRaises(Exception, csvinst.append_row(newrow, write=True))
        csvrows, header = CSVWrap.readCSV(Path(csv_path))
        self.assertEqual(len(csvrows), 55)
        self.assertEqual(len(header), 3)

if __name__ == '__main__':
    unittest.main()