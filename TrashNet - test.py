import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import random

test_data_dir = r"C:\Users\Eli\Desktop\test"
image_size = (224, 224)

model = tf.keras.models.load_model(r"C:\Users\Eli\AppData\Roaming\JetBrains\PyCharmCE2022.1\scratches"
                                   r"\trash_classifier_model")  # Load your trained model

image_files = os.listdir(test_data_dir)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

trash_images = [file for file in image_files if 'trash' in file.lower() and file.lower().endswith(image_extensions)]
non_trash_images = [file for file in image_files if 'trash' not in file.lower() and file.lower().endswith(image_extensions)]

if trash_images or non_trash_images:
    if trash_images:
        random_image = random.choice(trash_images)
        prediction_label = "contains trash"
    else:
        random_image = random.choice(non_trash_images)
        prediction_label = "does not contain trash"

    image_path = os.path.join(test_data_dir, random_image)
    print("Path to the chosen image:", image_path)

    image = load_img(image_path, target_size=image_size)
    image_array = img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    image_array /= 255.0

    prediction = model.predict(image_array)
    if prediction[0][0] < 0.5:
        print(f"The image {prediction_label}")
    else:
        print(f"The image {prediction_label}")
else:
    print("No image files found in the selected directory")
