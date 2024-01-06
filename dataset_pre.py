# Dataset preparation

import os
import pandas as pd
import shutil
import random
from sklearn.model_selection import train_test_split

QUERY_SIZE = 15

# EuroSAT
eu_source_path = './data/EuroSAT_RGB'
eurosat_path = './data/EuroSAT'

# The format of Plant Disease Dataset has been well prepared
# The format of Stanford Cars Dataset has been well prepared

# CUB
cub_path = './data/CUB_200_2011'
cub_images_path = os.path.join(cub_path, 'images')
cub_labels_file_path = os.path.join(cub_path, 'image_class_labels.txt')
cub_split_file_path = os.path.join(cub_path, 'train_test_split.txt')
cub_new_path = './data/CUB'

def init_EuroSAT():
    if os.path.exists(eurosat_path):
        return
    
    os.makedirs(eurosat_path)
    
    for class_folder in os.listdir(eu_source_path):
        class_path = os.path.join(eu_source_path, class_folder)

        image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

        train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

        train_path = os.path.join(eurosat_path, 'train', class_folder)
        test_path = os.path.join(eurosat_path, 'test', class_folder)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        for file_name in train_files:
            src_path = os.path.join(class_path, file_name)
            dest_path = os.path.join(train_path, file_name)
            shutil.copy(src_path, dest_path)

        for file_name in test_files:
            src_path = os.path.join(class_path, file_name)
            dest_path = os.path.join(test_path, file_name)
            shutil.copy(src_path, dest_path)
    
def init_CUB():
    if os.path.exists(cub_new_path):
        return
    
    os.makedirs(cub_new_path)

    split_dict = {}
    with open(cub_split_file_path, 'r') as split_file:
        for line in split_file:
            image_id, split_value = line.strip().split()
            split_dict[int(image_id)] = split_value

    labels_dict = {}
    with open(cub_labels_file_path, 'r') as labels_file:
        for line in labels_file:
            image_id, label = line.strip().split()
            labels_dict[int(image_id)] = label

    image_id = 1
    for class_folder in sorted(os.listdir(cub_images_path), key=lambda x: int(''.join(filter(str.isdigit, x)))):
        class_path = os.path.join(cub_images_path, class_folder)

        train_path = os.path.join(cub_new_path, 'train', class_folder)
        test_path = os.path.join(cub_new_path, 'test', class_folder)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        for image_file in sorted(os.listdir(class_path), key=lambda x: int(x.split('_')[-2])):
            image_label = labels_dict.get(image_id, None)
            image_split = split_dict.get(image_id, None)

            if image_label is not None and image_split is not None:
                image_path = os.path.join(class_path, image_file)

                if image_split == '1':
                    dest_path = os.path.join(train_path, image_file)
                else:
                    dest_path = os.path.join(test_path, image_file)

                shutil.copy(image_path, dest_path)
            else:
                print("error")
            image_id += 1
            
            
# Spilt data into n-way-k-shot-dataset
def split_shot_data(base_folder, train_combine_folder, test_combine_folder, n, images_per_class):
    if os.path.exists(train_combine_folder):
        #delete combine_folder
        shutil.rmtree(train_combine_folder)
    if os.path.exists(test_combine_folder):
        #delete combine_folder
        shutil.rmtree(test_combine_folder)
    
    os.makedirs(train_combine_folder)
    os.makedirs(test_combine_folder)
    
    # Get a list of n randomly selected class folders
    class_folders = random.sample(os.listdir(base_folder+"train"), n)
    
    for class_folder in class_folders:
        class_folder_path = os.path.join(base_folder+"train", class_folder)

        if os.path.isdir(class_folder_path):
            image_files = [f for f in os.listdir(class_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if image_files:
                # Create a folder for each class in num_shot_folder
                class_shot_folder = os.path.join(train_combine_folder, class_folder)
                os.makedirs(class_shot_folder, exist_ok=True)

                # Randomly choose images from the current class
                selected_images = random.sample(image_files, images_per_class)

                # Copy selected images to the num-shot folder
                for selected_image in selected_images:
                    source_path = os.path.join(class_folder_path, selected_image)
                    destination_path = os.path.join(class_shot_folder, f"{selected_image}")
                    shutil.copy(source_path, destination_path)

    print(f"{n}-way-{images_per_class}-shot dataset created successfully.")
    
    for class_folder in class_folders:
        class_folder_path = os.path.join(base_folder+"test", class_folder)

        if os.path.isdir(class_folder_path):
            image_files = [f for f in os.listdir(class_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if image_files:
                # Create a folder for each class in num_shot_folder
                class_shot_folder = os.path.join(test_combine_folder, class_folder)
                os.makedirs(class_shot_folder, exist_ok=True)

                # Randomly choose images from the current class
                selected_images = random.sample(image_files, QUERY_SIZE)

                # Copy selected images to the num-shot folder
                for selected_image in selected_images:
                    source_path = os.path.join(class_folder_path, selected_image)
                    destination_path = os.path.join(class_shot_folder, f"{selected_image}")
                    shutil.copy(source_path, destination_path)

    print(f"{n}-way-{QUERY_SIZE}-query dataset created successfully.")
    

    