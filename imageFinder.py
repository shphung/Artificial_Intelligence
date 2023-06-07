import os, shutil, sys, random, re, numpy as np
import matplotlib.pyplot as plt
import image2vect
"""
Name: Steven Phung
ID: 028023383
Course: CECS551-Advanced Artificial Intelligence
Assignment: 8
    Parts completed in this file:
        C: Downloaded img_celeba.7z and extracted
        D: using identity_CelebA.txt and selected_ids.txt to create selected dataset
        F: Implement imageFinger.py
        G: Compute precision and recall for different tau's
        H: Repeat f-g for 10 randomly selected unique celebrities. Draw figure
File Name: imageFinder.py : uses image2vect.py to compute Euclidean distance between images
Due:4/7/22
"""
# Initial command line argument I used for this file: (only takes 1 input)
# 000109.jpg

# C) & D) /img_celeba folder is in current working directory
# Function: create_selected_dataset()
# Read from text files to get selected IDs and create selected dataset
def create_selected_dataset():
    # Only if directory doesn't exist / directory is empty
    if (not os.path.exists(os.getcwd() + "\selected_dataset")) or (os.listdir(os.getcwd() + "\selected_dataset")) == 0:
        # Read from file and get IDs
        selected_ids = []
        file = open("selected_ids.txt", "r")
        lines = file.readlines()
        for i in lines:
            selected_ids.append(i.strip())
            
        # Read from file and isolate selected IDs and their respective file name
        selected_files = []
        file = open("identity_CelebA.txt", "r")
        lines = file.readlines()
        for i in lines:
            for j in selected_ids:
                if(i.strip().endswith(" " + j)):
                    s = i.strip().split(" ")
                    selected_files.append(s[0])
                    
        # Move data from original dataset to selected_dataset
        # Python's current directory with image dataset
        original_dataset_dir = os.getcwd() + "\img_celeba"
        selected_dataset_dir = os.getcwd() + "\selected_dataset"
        
        # Create directory if does not exist
        if not os.path.exists(selected_dataset_dir):
            os.mkdir(selected_dataset_dir)
            
        # Copy photos from "img_celeba" to "selected_dataset"
        for i in selected_files:
            src = original_dataset_dir + "\\" + i
            dst = selected_dataset_dir + "\\" + i
            shutil.copyfile(src, dst)
        print("Completed copying images from 'img_celeba' to 'selected_dataset'.")
    else:
        print("Skipping copy... selected_dataset directory already exists.")

# Function: euclidean_distance(vectorA, vectorB)
# Calculates the euclidean distance between two different dimensioned vectors
# Returns: Euclidean distance
def euclidean_distance(vectA, vectB):
    total = 0
    # vectA smaller than vectB
    if vectA.shape[0] < vectB.shape[0]:
        for i in range(len(vectB)):
            if i < len(vectA):
                total = (total + ((vectB[i] - vectA[i]) ** 2))
            else:
                total = (total + (vectB[i] ** 2))
    # vectA bigger than vectB
    else:
        for i in range(len(vectA)):
            if i < len(vectB):
                total = (total + ((vectA[i] - vectB[i]) ** 2))
            else:
                total = (total + (vectA[i] ** 2))
    return np.sqrt(total)

# F)
# Function: calculate_distance(initial_input, fileName)
# Initial input of an image of a celebrity from the selected_dataset. Using image2vect.py,
# Euclidean distance is calculated using the other images, resulting in 1,199 Euclidean distances.
def calculate_distances(initial_input, fileName):
    # Clear distances.txt
    with open(fileName, 'w') as myFile:
        myFile.write("")

    # Run image2vect on initial input and get corresponding embedding vector
    image2vect.run_yoloface("python yoloface/yoloface.py --model-cfg yoloface/cfg/yolov3-face.cfg --model-weights yoloface/model-weights/yolov3-wider_16000.weights --image selected_dataset/" + initial_input)
    vectA = image2vect.get_embedding_vector(initial_input)
    
    # For the other 1,199 files
    list_of_files = os.listdir("selected_dataset")
    for i in range(len(list_of_files)):
        print(str(i+1) + "/1199", end = " ")
        # Continue if current_file in directory matches initial_input_file
        if list_of_files[i] == initial_input:
            continue
        else:
            # Run image2vect on current_file and get corresponding embedding vector
            image2vect.run_yoloface("python yoloface/yoloface.py --model-cfg yoloface/cfg/yolov3-face.cfg --model-weights yoloface/model-weights/yolov3-wider_16000.weights --image selected_dataset/" + list_of_files[i])
            vectB = image2vect.get_embedding_vector(list_of_files[i])
            
            # Euclidean distance
            distance = euclidean_distance(vectA, vectB)
            
            # Write d(img1, img2) to a file, total of 1,199 in file
            with open(fileName, 'a') as myFile:
                 myFile.write(str(i+1) + ". d(" + initial_input + ", " + list_of_files[i] + "): " + str(distance) + "\n")

# Function: same_celebrity(resultsFile)
# The other images will be recognized as the same celebrity of the input image
# if the Euclidean distance is less than Tau.
# Return: list of images recognized as the same celebrity of the input image.
def same_celebrity(tau, resultsFile):
    same_celebs = []
    #Comparing tau to results
    file = open(resultsFile, "r")
    lines = file.readlines()
    for i in lines:
        distance = re.findall(r"[-+]?\d*\.\d+|\d+", i.split(":")[1])
        #If Euclidean distance < tau then image is same celebrity
        if float(distance[0]) < tau:
            same_celebs.append(i.strip())
    return same_celebs

# Function: precision_recall(same_celebs, resultsFile)
# Calculate precision and recall
# Returns: celeb_id, precision, and recall
def precision_recall(celeb_photo, same_celebs, resultsFile):
    if len(same_celebs) == 0:
        return celeb_photo, 0, 0
    number_of_correctly_recognized_images = 0
    file = open("identity_CelebA.txt", "r")
    file_lines = file.readlines()
    for j in file_lines:
        if(j.strip().startswith(celeb_photo)):
            celeb_id = j.strip().split(" ")[1]
    
    # If processed image matches celeb_id then picture was recognized correctly
    for i in range(len(same_celebs)):
        #print(str(i) + " " + same_celebs[i])
        second_file = same_celebs[i].split(")")[0]
        second_file = second_file[-10:]
        for j in file_lines:
            if j.strip().startswith(second_file) and j.strip().endswith(celeb_id):
                number_of_correctly_recognized_images = number_of_correctly_recognized_images + 1
    
    # Precision / Recall
    precision = number_of_correctly_recognized_images / len(same_celebs)
    recall = number_of_correctly_recognized_images / 40
    return celeb_id, precision, recall

        
# G)
# Function: generate_random_celebs()
# Goes through selected_dataset, grabs first unique instance of every celebrity
# Returns: random sample of 9 unique celebrities
def generate_random_celebs(initial_input):
    # Read from file and get IDs
    selected_ids = []
    file = open("selected_ids.txt", "r")
    lines = file.readlines()
    for i in lines:
        selected_ids.append(i.strip())
    
    # Read from file and isolate selected IDs and their respective file name
    selected_files = []
    file = open("identity_CelebA.txt", "r")
    lines = file.readlines()
    
    # Remove initial input from list to keep unique-ness
    rejected_id = ""
    for i in selected_ids:
        for j in lines:
            if(j.strip().startswith(initial_input + " ")):
                rejected_id = i
    selected_ids.remove(rejected_id)
    
    # Find first instance of each unique celebrity (40 total)
    for i in selected_ids:
        for j in lines:
            if(j.strip().endswith(" " + i)):
                selected_files.append(j.strip().split(" ")[0])
                break
        else:
            continue
        continue
    
    # Return random sample
    return random.sample(selected_files, 9)

# Function: main()
def _main():
    # Create selected_dataset directory
    create_selected_dataset()
    # Generate 9 random, unique celeb_ids (excluding initial input from command line)
    random_unique_celebs = generate_random_celebs(str(sys.argv[1]))
    # Insert input into list
    random_unique_celebs.insert(0, str(sys.argv[1]))
    # Tau intervals
    tau = [0, 0.5, 1, 1.5, 5, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]

    with open('final_results.txt', 'w') as myFile:
        myFile.write("")

    # Repeating steps F) and G) for 10 different unique celebrities
    for i in range(0,10):
        print("Iteration", str(i+1) + ":", end = "\n")
        calculate_distances(random_unique_celebs[i], "results_" + str(i+1) + ".txt")
        for j in range(len(tau)):
            same_celebs = same_celebrity(tau[j], "results_" + str(i+1) + ".txt")
            celeb_id, precision, recall = precision_recall(random_unique_celebs[i], same_celebs,  "results_" + str(i+1) + ".txt")
            with open('final_results.txt', 'a') as myFile:
                myFile.write(str(celeb_id) + " " + str(tau[j]) + " " + str(precision) + " " + str(recall) + "\n")

    # H) Draw the overlapped precision and recall curves
    file = open("final_results.txt", "r")
    lines = file.readlines()
    
    celeb_id = []
    for i in range(1, len(lines), 21):
        celeb_id.append(lines[i].strip().split(" ")[0])
    
    colors = []
    for i in range(0, 10):
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    
    plt.title("Precision Curve")
    for i in range(0, 10):
        precision = []
        for j in range(i * 21, (i+1) * 21):
            precision.append(float(lines[j].strip().split(" ")[2]))
        plt.plot(tau, precision, c=colors[i], label=celeb_id[i])
    plt.xlabel('Tau')
    plt.ylabel('Precision')
    plt.xticks(np.arange(0, 10, .5))
    plt.yticks(np.arange(0, 0.10, 0.01))
    plt.legend()
    plt.show()
    
    plt.clf()
    plt.title("Recall Curve")
    for i in range(0, 10):
        recall = []
        for j in range(i * 21, (i+1) * 21):
            recall.append(float(lines[j].strip().split(" ")[3]))
        plt.plot(tau, recall, c=colors[i], label=celeb_id[i])
    plt.xlabel('Tau')
    plt.ylabel('Recall')
    plt.xticks(np.arange(0, 10, .5))
    plt.yticks(np.arange(0, 0.5, 0.1))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    _main()





































