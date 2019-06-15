from sklearn.model_selection import train_test_split

import os
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
import multiprocessing as mp

def resize_image_square(image_path, size=299):
    try:
        pil_image = Image.open(image_path)
        width, height = pil_image.size   # Get dimensions
        if width != height:
            min_size = min(width, height)

            left = (width - min_size)/2
            top = (height - min_size)/2
            right = (width + min_size)/2
            bottom = (height + min_size)/2

            cropped = pil_image.crop((left, top, right, bottom))
            resized = cropped.resize((size,size))
        else:
            resized = pil_image.resize((size,size))

        return image_path, resized
    except:
        with open("error_images.txt", "a") as w:
            w.write(image_path + "\n")
        return image_path, None

if __name__ == "__main__":
    print("Loading images...")
    IMAGE_SIZE = 64
    output_directory_name = "all_images_resized_{}".format(IMAGE_SIZE)
    # images = glob("/hdd/kaggle/landmarks/all_images_resized/*.jpg")
    images = glob("/home/daniel/kaggle/landmarks/all_images_resized_299/*.jpg")
    already_resized = glob("/home/daniel/kaggle/landmarks/{}/*.jpg".format(output_directory_name))

    already_resized_dict = {key.split("/")[-1]: "" for key in already_resized}

    need_to_process = [image for image in images if image.split("/")[-1] not in already_resized_dict]

    print("Already processed: {:.3f}".format(1 - len(need_to_process)/ len(images)))
    print("Finished loading images.")

    del images
    del already_resized
    del already_resized_dict
    images = need_to_process
    del need_to_process

    errors = 0

    pool = mp.Pool(processes=6)
    outdir = "/home/daniel/kaggle/landmarks/{}/".format(output_directory_name)

    for image_path, resized in tqdm(pool.imap_unordered(resize_image_square, images), total=len(images)):
        if resized == None:
            errors += 1
            continue
        img_name = image_path.split("/")[-1]
        resized.save(outdir + img_name, "JPEG")


    print("Total errors: {}, Percent errors: {:.4f}".format(errors, errors / len(images)))