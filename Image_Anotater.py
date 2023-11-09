import cv2
import PIL
import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import torchvision.transforms as T
import os
from tqdm import tqdm

def annotate_images(source_folder, output_folder):
    # Get the folder name without the extension
    folder_name = os.path.basename(os.path.normpath(source_folder))

    # Create the output folder if it doesn't exist
    output_folder = os.path.join(output_folder, folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the source folder
    file_list = [f for f in os.listdir(source_folder) if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(source_folder, f))]

    # Print the list of filenames
    for filename in tqdm(file_list, desc="Now Annotating Images..."):
        try:
            image = cv2.imread(os.path.join(source_folder, filename))

            # Check if the image is None (i.e., not read successfully)
            if image is None:
                print(f"Error reading {filename}. Skipping.")
                continue

            # Convert the image to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the edges of the water bottle objects using Canny edge detection
            edges = cv2.Canny(grayscale_image, 50, 150)

            # Find the contours of the water bottle objects
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw a rectangle around each water bottle object
            rectangle_color = (0, 0, 255)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x+w, y+h), rectangle_color, 2)

            # Save the modified image to the subdirectory
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)
            # print(f"image file {filename} written sucessfully")

        except cv2.error as e:
            print(f"Error processing {filename}: {e}")
            continue  # Skip to the next iteration if an error occurs

# torch.transforms

# grayscale
grayscale_transform = T.Grayscale(3)

# random rotation
random_rotation_transformation_45 = T.RandomRotation(45)
random_rotation_transformation_85 = T.RandomRotation(85)
random_rotation_transformation_65 = T.RandomRotation(65)

# Gaussian Blur
gaussian_blur_transformation_13 = T.GaussianBlur(kernel_size=(7, 13), sigma=(6, 9))
gaussian_blur_transformation_56 = T.GaussianBlur(kernel_size=(7, 13), sigma=(5, 8))

# Gaussian Noise
def addnoise(input_image, noise_factor=0.3):
    inputs = T.ToTensor()(input_image)
    noisy = inputs + torch.rand_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0, 1.)
    output_image = T.ToPILImage()
    image = output_image(noisy)
    return image

# Colour Jitter
colour_jitter_transformation_1 = T.ColorJitter(brightness=(0.5, 1.5), contrast=(3), saturation=(0.3, 1.5), hue=(-0.1, 0.1))
colour_jitter_transformation_2 = T.ColorJitter(brightness=(0.7), contrast=(6), saturation=(0.9), hue=(-0.1, 0.1))
colour_jitter_transformation_3 = T.ColorJitter(brightness=(0.5, 1.5), contrast=(2), saturation=(1.4), hue=(-0.1, 0.5))

# Random invert
random_invert_transform = T.RandomInvert()

# Main function that calls all the above functions to create 11 augmented images from one image
def augment_image(img_path):
    # orig_image
    orig_img = Image.open(Path(img_path))

    # grayscale
    grayscaled_image = grayscale_transform(orig_img)

    # random rotation
    random_rotation_transformation_45_image = random_rotation_transformation_45(orig_img)
    random_rotation_transformation_85_image = random_rotation_transformation_85(orig_img)
    random_rotation_transformation_65_image = random_rotation_transformation_65(orig_img)

    # Gaussian Blur
    gaussian_blurred_image_13_image = gaussian_blur_transformation_13(orig_img)
    gaussian_blurred_image_56_image = gaussian_blur_transformation_56(orig_img)

    # Gaussian Noise
    gaussian_image_3 = addnoise(orig_img)
    gaussian_image_6 = addnoise(orig_img, 0.6)
    gaussian_image_9 = addnoise(orig_img, 0.9)

    # Color Jitter
    colour_jitter_image_1 = colour_jitter_transformation_1(orig_img)
    colour_jitter_image_2 = colour_jitter_transformation_2(orig_img)
    colour_jitter_image_3 = colour_jitter_transformation_3(orig_img)

    return [
        orig_img, grayscaled_image, random_rotation_transformation_45_image, random_rotation_transformation_65_image,
        random_rotation_transformation_85_image, gaussian_blurred_image_13_image, gaussian_blurred_image_56_image,
        gaussian_image_3, gaussian_image_6, gaussian_image_9, colour_jitter_image_1, colour_jitter_image_2,
        colour_jitter_image_3
    ]

# augmented_images = augment_image(orig_img_path)

def creating_file_with_augmented_images(source_folder, output_folder):
    # Get the folder name without the extension
    folder_name = os.path.basename(os.path.normpath(source_folder))

    # Create the output folder if it doesn't exist
    output_folder = os.path.join(output_folder, folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the source folder
    file_list = [f for f in os.listdir(source_folder) if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(source_folder, f))]
    counter = 0
    for image in tqdm(file_list, desc="Now Augmenting Images..."):
                counter += 1
                required_images = augment_image(os.path.join(source_folder, image))
                
                counter2 = 0
                for augmented_image in required_images:
                    counter2 += 1
                    counter3 = 0
                    augmented_image = augmented_image.save(
                        f"{output_folder}/{counter}_{counter2}_{counter3}_{image}"
                    )

if __name__ == "__main__":
    while True:
         # Menu
         print("Menu:")
         print("1. Annotate Images")
         print("2. Augment Images")
         print("3. Annotate and Augment Images")
         print("4. Exit")
        
         choice = input("Enter your choice (1/2/3/4): ")

         if choice == "1":
            source_folder = input("Enter the path of the images to annotate and/or augment: ")
            folder_name = os.path.basename(os.path.normpath(source_folder))
            # Annotate Images
            output_folder = input("Enter the path where you want to save the annotated images: ")
            annotated_folder = output_folder+ "/" + folder_name +"_annotated/"
            annotate_images(source_folder, annotated_folder)
            print(f"Annotated images saved in: {annotated_folder}")

         elif choice == "2":
            source_folder = input("Enter the path of the images to annotate and/or augment: ")
            folder_name = os.path.basename(os.path.normpath(source_folder))
            # Augment Images
            output_folder = input("Enter the path where you want to save the augmented images: ")
            creating_file_with_augmented_images(source_folder, output_folder)
            print(f"Augmented images saved in: {output_folder}")

         elif choice == "3":
            source_folder = input("Enter the path of the images to annotate and/or augment: ")
            folder_name = os.path.basename(os.path.normpath(source_folder))
            output_folder = input("Enter the path where you want to save the result of images: ")
            annotate_images(source_folder, source_folder+ "/" + folder_name +"_annotated/")
            creating_file_with_augmented_images(source_folder+ "/" + folder_name +"_annotated/"+ folder_name, output_folder)
         elif choice == "4":
             break
         else:
             print("Invalid choice. Please enter 1, 2, or 3.")
