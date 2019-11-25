import argparse
import io

from google.cloud import storage
from google.cloud import vision
import json

def detect_text(args):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(args.img, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)

    text_arr = []

    for text in response.text_annotations:
        text_obj = {}
        text_obj["description"] = text.description
        text_obj["bounds"] = []
        for vertex in text.bounding_poly.vertices:
            bound_obj = {}
            bound_obj["x"] = vertex.x
            bound_obj["y"] = vertex.y
            text_obj["bounds"].append(bound_obj)

        text_arr.append(text_obj)
    
    # print("Writing:", args.out)
    with open(args.out, "w") as f:
        f.write(json.dumps(text_arr))
    

parser = argparse.ArgumentParser(description='U network for segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--img', type=str, help='Input Image', required=True)
parser.add_argument('--out', type=str, help='Output text', default="./out.json")

args = parser.parse_args()

try:
    detect_text(args)
except Exception as e:
    print(e)
    print("Set environment variable 'export GOOGLE_APPLICATION_CREDENTIALS=/path/to/json/with_credentials.json' documentation at https://cloud.google.com/docs/authentication/getting-started") 