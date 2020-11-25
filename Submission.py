# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 08:13:59 2020

@author: vasan
"""

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy.random import seed

seed(1)
import os
os.chdir("D:/Python-deep learning keras/Interns Assignment-20201118T003506Z-001/Interns Assignment/")

#remove files that are corrupted
skip=0
for folder_name in ("1 root","2 or more roots"):
    folder_path=os.path.join("xrays-database",folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path,fname)
        try:
            fobj=open(fpath, 'rb')
            is_jfif=tf.compat.as_bytes("JFIF") in fobj.peek(10)
            
        finally:
            fobj.close()
        if not is_jfif:
            skip +=1
            os.remove(fpath)
            
print("Deleted images %d" % skip)

kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])

#sharpening the images
for folder_name in ("1 root","2 or more roots"):
    folder_path=os.path.join("xrays-database",folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path,fname)
        sharpened = cv2.filter2D(cv2.imread(fpath), -1, kernel)
        cv2.imwrite(fname, sharpened)
        
image_size =(230,130)
batch_size = 25

train_ds =tf.keras.preprocessing.image_dataset_from_directory(
        "xrays-database",
        validation_split=0.3,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
)

class_names = train_ds.class_names
print(class_names)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "xrays-database",
        validation_split=0.3,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
)

#visualize the data
plt.figure(figsize=(10,10))
for images,labels in train_ds.take(1):
    for i in range(9):
        ax =plt.subplot(3,3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
        
#image representation
data_augmentation = keras.Sequential(
        [
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomFlip("vertical"),
                ])
    
plt.figure(figsize=(10,10))
for images,_ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax=plt.subplot(3, 3 ,i+1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

train_ds=train_ds.prefetch(buffer_size = 25)
val_ds = val_ds.prefetch(buffer_size =25)

def k_model(input_shape, num_classes):
    inputs=keras.Input(shape= input_shape)
    x=data_augmentation(inputs)
    
    #x=layers.experimental.preprocessing.Rescaling(1.0 /255)(x)
    x=layers.Conv2D(32, kernel_size = (3,3), activation = 'relu')(x)
    x=layers.BatchNormalization()(x)
    
    x=layers.MaxPooling2D(pool_size=(2,2))(x)
    x=layers.Conv2D(64, kernel_size = (3,3), activation = 'relu')(x)
    x=layers.BatchNormalization()(x)
    
    
    x=layers.MaxPooling2D(pool_size = (2,2))(x)
    x=layers.Flatten()(x)
    
      
    x = layers.Dropout(0.5)(x)

    if num_classes == 2:
        activation = "sigmoid"
        units=1
    else:
        activation = "softmax"
        units = num_classes
        
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units,activation=activation)(x)
    return keras.Model(inputs, outputs)





model = k_model(input_shape=image_size+(3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)

epochs=45

callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5", monitor = "val_accuracy", save_best_only = True),
]

model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
)

model.fit(
        train_ds,epochs=epochs,callbacks=callbacks,validation_data=val_ds,
)


img = keras.preprocessing.image.load_img(
    "D:/Python-deep learning keras/Interns Assignment-20201118T003506Z-001/Interns Assignment/xrays-database/2 or more roots/tooth1.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(
    "This image is %.2f percent 1 root and %.2f percent 2 or more."
    % (100 * (1 - score), 100 * score)
)