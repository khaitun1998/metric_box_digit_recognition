"""
testReader.py

Run: python testReader.py --input_folder input_folder --output_file output.csv --data_folder data

Description: init a reader with data folder then process all images from input_folder,
write the results to output.csv with header (filename,number)
"""

import sys
import os
import time

# version 4.1.1.26
import cv2

# Import reader
from readingMeter import Reader


def process(input_img):
    reader = Reader()
    reader.prepare()

    img = cv2.imread(input_img)
    number = reader.process(img)
    return number


if __name__=="__main__":

    # Get input parameters
    input_img = sys.argv[1]

    # Run time
    print(process(input_img))

