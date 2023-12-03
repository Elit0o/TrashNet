import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random

train_data_dir = r"C:\Users\Eli\Desktop\train"
image_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_data = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    input_tensor=tf.keras.layers.Input(shape=(224, 224, 3), name='input_image')
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)

epochs = 10
model.fit(
    train_data,
    epochs=epochs,
    validation_data=validation_data
)

model.save("trash_classifier_model")

directory_paths = [
    r"C:\Users\Eli\Desktop\train\0",  # Directory for images containing trash
     r"C:\Users\Eli\Desktop\train\1"   # Directory for images not containing trash
]

random_directory = random.choice(directory_paths)
files = os.listdir(random_directory)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

image_files = [file for file in files if file.lower().endswith(image_extensions)]

if image_files:
    random_image = random.choice(image_files)
    image_path = os.path.join(random_directory, random_image)
    print("Path to the chosen image:", image_path)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)  # Expand dimensions to match batch size

    image_array /= 255.0

    prediction = model.predict(image_array)
    if prediction[0][0] < 0.5:
        print("The image contains trash")
    else:
        print("The image does not contain trash")
else:
    print("No image files found in the selected directory")