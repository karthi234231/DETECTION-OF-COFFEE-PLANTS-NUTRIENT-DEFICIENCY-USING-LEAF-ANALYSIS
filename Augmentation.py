import os
import cv2
import imgaug.augmenters as iaa
import numpy as np

dataset_path = r"C:\Users\KARTHIK\Desktop\CoLeaf DATASET"
target_count = 200  # Desired number of images per category

augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),  
    iaa.Affine(rotate=(-25, 25)),  
    iaa.Multiply((0.8, 1.2)),  
])

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.isdir(folder_path):
        continue

    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    current_count = len(images)

    if current_count == 0:
        print(f"Warning: No images found in {folder}. Skipping augmentation.")
        continue  # Skip this folder

    print(f"Augmenting {folder}: Adding {target_count - current_count} images.")

    for i in range(current_count, target_count):
        img_name = images[i % current_count]  # Avoid division by zero
        img_path = os.path.join(folder_path, img_name)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable file: {img_path}")
            continue  # Skip unreadable files

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented_img = augmenters(image=img)

        new_img_name = f"{folder}({i+1}).jpg"
        cv2.imwrite(os.path.join(folder_path, new_img_name), cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))

print("Augmentation completed successfully!")