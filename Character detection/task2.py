"""
Character Detection

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os

import utils
import task1  # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def detect(img, template):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # TODO: implement this function.
    R=len(img)
    C=len(img[0]) 
    tr=len(template)
    tc=len(template[0])
    coordinates=[]
    
   
    
    for i in range(tr,R,1):
        for j in range(tc,C,1):
            imgcropped= utils.crop(img, i-tr, i, j-tc, j)
            normimg= task1.normalize(imgcropped)
            
            #inverting the normalized image and setting the values to 1 and 0 only
            for a in range(tr):
                for b in range(tc):
                    if normimg[a][b]>=0.45:
                        normimg[a][b]=0
                    else:
                        normimg[a][b]=1
            
            normtemp= task1.normalize(template)
            
            for p in range(tr):
                for q in range(tc):
                    if normtemp[p][q]>=0.7:
                        normtemp[p][q]=0
                    else:
                        normtemp[p][q]=1
           
            
            #using SSD to match template   
            diff= utils.elementwise_sub(normimg, normtemp)
            imgsqr= utils.elementwise_mul(diff, diff)
           
            ssdsum=[]
            for a in imgsqr:
                ssdsum.append(sum(a))
            z= sum(ssdsum)
            if (z < 12):
                tup= (i-tr+1,j-tc+1)
                coordinates.append(tup)
               
                del tup
            del ssdsum    
                
    
    
    #raise NotImplementedError
    return coordinates




def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = task1.read_image(args.img_path)
    template = task1.read_image(args.template_path)

    coordinates = detect(img, template)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()
