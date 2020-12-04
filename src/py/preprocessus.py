
from pathlib import Path
import numpy as np
from FamliImgIO import dcmio

def extractImageArrayFromUS(file_path, out_dir=None, rescale_size=None):

    dcmobj = dcmio.DCMIO(file_path)
    # Create the output path
    out_path = None
    if out_dir is not None:
        out_path = out_dir / (file_path.stem + '.jpg')
    
    np_frame = dcmobj.get_repres_frame(out_path)
    us_type = dcmobj.get_type()
    us_model = dcmobj.get_model()

    return np_frame, us_type, us_model
    