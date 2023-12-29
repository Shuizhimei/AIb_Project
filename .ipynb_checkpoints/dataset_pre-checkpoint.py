# Dataset preparation

import os
import pandas as pd
import shutil
import random


# ChestX
cx_label_path = './data/Coronahack-Chest-XRay-Dataset/Chest_xray_Corona_Metadata.csv'
cx_test_path = './data/Coronahack-Chest-XRay-Dataset/test'
cx_train_path = './data/Coronahack-Chest-XRay-Dataset/train'
cx_output_path = './data/chestX/'

# Preprocess ChestX Dataset
def init_chestX():
    if os.path.exists(cx_output_path):
        return
    
    # 读取标签文件
    labels_df = pd.read_csv(cx_label_path)

    # 创建输出文件夹
    os.makedirs(os.path.join(cx_output_path, 'train', 'Pnemonia'), exist_ok=True)
    os.makedirs(os.path.join(cx_output_path, 'train', 'Normal'), exist_ok=True)
    os.makedirs(os.path.join(cx_output_path, 'test', 'Pnemonia'), exist_ok=True)
    os.makedirs(os.path.join(cx_output_path, 'test', 'Normal'), exist_ok=True)

    # 分割训练数据
    for index, row in labels_df.iterrows():
        image_name = row['X_ray_image_name']
        label = row['Label']
        category = row['Dataset_type']
        if category == "TRAIN":
            source_path = os.path.join(cx_train_path, image_name)
            destination_path = os.path.join(cx_output_path, 'train', label, image_name)
        else:
            source_path = os.path.join(cx_test_path, image_name)
            destination_path = os.path.join(cx_output_path, 'test', label, image_name)
        shutil.copyfile(source_path, destination_path)

# Spilt data into k-shot-dataset
def split_shot_data(base_folder, combine_folder, images_per_class):
    if os.path.exists(combine_folder):
        return
    
    os.makedirs(combine_folder)

    for class_folder in os.listdir(base_folder):
        class_folder_path = os.path.join(base_folder, class_folder)

        if os.path.isdir(class_folder_path):
            image_files = [f for f in os.listdir(class_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if image_files:
                # Create a folder for each class in num_shot_folder
                class_shot_folder = os.path.join(combine_folder, class_folder)
                os.makedirs(class_shot_folder, exist_ok=True)

                # Randomly choose images from the current class
                selected_images = random.sample(image_files, images_per_class)

                # Copy selected images to the num-shot folder
                for selected_image in selected_images:
                    source_path = os.path.join(class_folder_path, selected_image)
                    destination_path = os.path.join(class_shot_folder, f"{selected_image}")
                    shutil.copy(source_path, destination_path)

    print(f"{images_per_class}-shot dataset created successfully.")
    