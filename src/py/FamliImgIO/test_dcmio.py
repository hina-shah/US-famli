import unittest
import numpy as np
from dcmio import DCMIO

class DCMIOest(unittest.TestCase):
    def test(self):
        dcmobj = DCMIO('/Users/hinashah/UNCFAMLI/Data/test/test.dcm')
        self.assertEqual(dcmobj.get_model(), 'V830')
        self.assertEqual(dcmobj.get_type(), 'cine')
        self.assertEqual(dcmobj.get_photometric_interpretation(), 'YBR_FULL_422')
        # read the image,
        # Get the middle frame and write it out
        img = dcmobj.get_repres_frame('/Users/hinashah/UNCFAMLI/Data/test/test.jpg')
        size = img.shape
        self.assertEqual(size[0], 735)
        self.assertEqual(size[1], 975)

if __name__ == '__main__':
    unittest.main()