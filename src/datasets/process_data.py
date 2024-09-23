

import os
import random
import shutil

# create 3 folder Mouse, Cat, Dog in side each folder src/datasets/train and src/datasets/val
def create_folders():
    # create folders
    os.makedirs("train/Mouse", exist_ok=True)
    os.makedirs("train/Cat", exist_ok=True)
    os.makedirs("train/Dog", exist_ok=True)
    os.makedirs("val/Mouse", exist_ok=True)
    os.makedirs("val/Cat", exist_ok=True)
    os.makedirs("val/Dog", exist_ok=True)

# randomly assign images from src/datasets/train to the classes Mouse, Cat, Dog
# just need 100 images for each class
def assign_images_train():
    # # list all images
    # images = os.listdir("train")
    # # shuffle images
    # random.shuffle(images)
    # # random 300 images
    # images = images[:300]
    # # assign images to classes
    # for i in range(len(images)):
    #     if i % 3 == 0:
    #         shutil.move("train/" + images[i], "train/Mouse")
    #     elif i % 3 == 1:
    #         shutil.move("train/" + images[i], "train/Cat")
    #     else:
    #         shutil.move("train/" + images[i], "train/Dog")
    # bring all images from each subfolder to the parent folder
    for folder in ["Mouse", "Cat", "Dog"]:
        images = os.listdir("src/datasets/train/" + folder)
        for image in images:
            shutil.move("src/datasets/train/" + folder + "/" + image, "src/datasets/train/" + image)

# randomly assign images from src/datasets/val to the classes Mouse, Cat, Dog
def assign_images_val():
    # # list all images
    # images = os.listdir("val")
    # # shuffle images
    # random.shuffle(images)
    # # random 300 images
    # images = images[:300]
    # # assign images to classes
    # for i in range(len(images)):
    #     if i % 3 == 0:
    #         shutil.move("val/" + images[i], "val/Mouse")
    #     elif i % 3 == 1:
    #         shutil.move("val/" + images[i], "val/Cat")
    #     else:
    #         shutil.move("val/" + images[i], "val/Dog")
    # bring all images from each subfolder to the parent folder
    for folder in ["Mouse", "Cat", "Dog"]:
        images = os.listdir("src/datasets/val/" + folder)
        for image in images:
            shutil.move("src/datasets/val/" + folder + "/" + image, "src/datasets/val/" + image)

# main function
def main():
    # create_folders()
    assign_images_train()
    assign_images_val()
    print("Data processed successfully!")

if __name__ == "__main__":
    main()