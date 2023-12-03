import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import random

test_data_dir = #data dir
image_size = (224, 224)

model = tf.keras.models.load_model( # Load your trained model) 

image_files = os.listdir(test_data_dir)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

valid_images = [file for file in image_files if file.lower().endswith(image_extensions)]

if valid_images:
    random_image = random.choice(valid_images)
    image_path = os.path.join(test_data_dir, random_image)
    print("Path to the chosen image:", image_path)

    image = load_img(image_path, target_size=image_size)
    image_array = img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    image_array /= 255.0

    prediction = model.predict(image_array)
    if prediction[0][0] < 0.5:
        print("The image contains trash")
    else:
        print("The image does not contain trash")
else:
    print("No valid image files found in the selected directory")
