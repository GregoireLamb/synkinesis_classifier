import os

import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

class ImagePreprocessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        register_heif_opener()

    def preprocess_images(self):

        if not os.path.exists(self.output_folder + "/train/"):
            os.makedirs(self.output_folder + "/train/true/")
            os.makedirs(self.output_folder + "/train/false/")
        if not os.path.exists(self.output_folder + "/validation/"):
            os.makedirs(self.output_folder + "/validation/true/")
            os.makedirs(self.output_folder + "/validation/false/")
        if not os.path.exists(self.output_folder + "/test/"):
            os.makedirs(self.output_folder + "/test/true/")
            os.makedirs(self.output_folder + "/test/false/")

        print(f"Input folder t : {self.input_folder}")
        patient_id = 0

        for true_false_folders in os.listdir(self.input_folder):
            print(f"Processing folder: {true_false_folders}")
            true_false = "true/" if true_false_folders == "with Synkinesia" else "false/"

            if true_false == "true/":
                for patient_folder in os.listdir(os.path.join(self.input_folder, true_false_folders)):
                    rnd = np.random.rand()
                    train_val_test = "train/" if rnd < 0.7 else "validation/" if rnd < 0.8 else "test/"
                    for file in os.listdir(os.path.join(os.path.join(self.input_folder, true_false_folders), patient_folder)):
                        image_path = os.path.join(os.path.join(os.path.join(self.input_folder, true_false_folders), patient_folder), file)
                        image = Image.open(image_path).convert('RGB')
                        image = self.crop_image(image)
                        image = self.resize_image(image)
                        output_path = os.path.join(self.output_folder, train_val_test + true_false + str(patient_id) + file.split(' ')[-1].split('.')[0].lower() + ".jpg")
                        image.save(output_path, "jpeg")
                    patient_id += 1
            else:
                for cat_folder in os.listdir(os.path.join(self.input_folder, true_false_folders)):
                    cat = "toset"
                    if cat_folder == "Closed eyes":
                        cat = "eyes"
                    elif cat_folder == "Full effort smile":
                        cat = "smile"
                    elif cat_folder == "Resting face":
                        cat = "rest"

                    for file in os.listdir(os.path.join(os.path.join(self.input_folder, true_false_folders), cat_folder)):
                        rnd = np.random.rand()
                        train_val_test = "train/" if rnd < 0.7 else "validation/" if rnd < 0.8 else "test/"

                        image_path = os.path.join(os.path.join(os.path.join(self.input_folder, true_false_folders), cat_folder), file)
                        image = Image.open(image_path).convert('RGB')
                        image = self.crop_image(image)
                        image = self.resize_image(image)
                        output_path = os.path.join(self.output_folder, train_val_test + true_false + str(patient_id) + cat + ".jpg")
                        image.save(output_path, "jpeg")
                        patient_id += 1

    def resize_image(self, image):
        # Resize the image to 256x256
        image = image.resize((256, 256))
        return image

    def crop_image(self, image):
        width, height = image.size
        if width > height:
            left = (width - height) / 2
            right = (width + height) / 2
            top = 0
            bottom = height
        else:
            left = 0
            right = width
            top = (height - width) / 2
            bottom = (height + width) / 2
        image = image.crop((left, top, right, bottom))
        return image

