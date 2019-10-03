
from pathlib import Path
import preprocess_us
import pytesseract
import csv
from matplotlib import pyplot as plt

def extractTagFromFrame(np_frame, bounding_box):

    sub_image = np_frame[bounding_box[0][1] : bounding_box[1][1],
                         bounding_box[0][0] : bounding_box[1][0] ]

    print(sub_image.shape)
    data = pytesseract.image_to_data(sub_image)
    print(data)

    plt.figure()
    plt.imshow(sub_image, cmap='gray', vmin=0, vmax=255)
    plt.show()

    # TODO: get the data as a dictionary and go from there instead of splitlines
    lines = data.splitlines()
    t = [line.split() for line in lines]
    conf_index = t[0].index('conf')
    conf = []
    for index in range(1,len(t)):
        conf.append(int(t[index][conf_index]))
    
    final_tag = 'Undecided'
    if max(conf) > 70:
        max_conf_tag_ind = conf.index(max(conf))+1
        tag_ind = t[0].index('text')
        if tag_ind < len(t[max_conf_tag_ind]):
            final_tag = t[max_conf_tag_ind][tag_ind]
    
    print(final_tag)
    return final_tag

data_folder = Path("/Users/hinashah/UNCFAMLI/Data/")
subjects = ["UNC-0414-1_20190830_111550", "UNC-0414-2_20190924_085140"]
out_images_dir = data_folder / 'Images_0' 
rescale_size = [800,600]
#tag_bounding_box = [[0,60], [200,150]]
tag_bounding_box = { 'V830':[[40,75], [255,190]],
                    'LOGIQe':  [[0,55], [200,160]]}

non_tag_us = ['Unknown', 'ge kretz image', 'Secondary capture image report',
                'Comprehensive SR', '3D Dicom Volume']

preprocess_us.checkDir(out_images_dir, False)

for subject_id in subjects:

    out_dir = out_images_dir / (subject_id + "_images")
    preprocess_us.checkDir(out_dir)
    
    csv_file = open(str(out_dir/ 'info.csv'), 'w')
    field_names = ['File', 'type', 'tag']
    csvfilewriter = csv.DictWriter(csv_file, field_names)
    csvfilewriter.writeheader()
    
    i=1
    file_names = list( (data_folder / subject_id).glob('**/*.dcm') )
    
    for file_name in file_names:

        print("============ FILE {} ==================== ".format(i))
        print(str(file_name))
        np_frame, us_type, capture_model = preprocess_us.extractImageArrayFromUS(file_name, out_dir, None)
        print(us_type)
        # Extract text from the image

        tag = 'Unknown'
        if us_type not in non_tag_us:
            tag = extractTagFromFrame(np_frame, tag_bounding_box[capture_model])
        
        csvfilewriter.writerow({'File': str(file_name), 'type': us_type, 'tag': tag})

        i+=1
    csv_file.close()

print('---- DONE ----')