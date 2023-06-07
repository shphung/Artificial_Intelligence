import os, numpy as np
from PIL import Image
"""
Name: Steven Phung
ID: 028023383
Course: CECS551-Advanced Artificial Intelligence
Assignment: 8
    Parts completed in this file:
        A: using open source YOLOFace
        B: imported YOLOFace
        E: used YOLOFace to implement image2vect.py
File Name: image2vect.py:
    Reads input of image file with a human face, crops the input image for the
    bounding box, and outputs embedding vector of said image.
    Only works on one image at a time
Due:4/7/22
"""
# A) & B)
# YOLOFace
def run_yoloface(args):
    os.system(args)

# E)
# Function: get_embedding_vector()
# Gets original image, crops based on bounding box, and then
# Returns: embedding vector based on cropped image.
def get_embedding_vector(fileName):
    # I added:
    # with open('results.txt', 'w') as myFile:
    #     myFile.write(str(left) + " " + str(top) + " " + str(right) + " " + str(bottom))
    # to the beginning of the draw_predict function in yoloface's utils.py in order to get the coordinates of the bounding box
    # Get coordinates from results.txt
    file = open("results.txt", "r")
    lines = file.readline()
    
    # Open original image and crop based on coordinates from results.txt
    im = Image.open("img_celeba/" + fileName)
    cropped = im.crop((int(lines.strip().split(" ")[0]), int(lines.strip().split(" ")[1]), int(lines.strip().split(" ")[2]), int(lines.strip().split(" ")[3])))
    
    # Create embedding vector based on cropped image
    image = np.array(cropped)
    embedding_vector = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return embedding_vector