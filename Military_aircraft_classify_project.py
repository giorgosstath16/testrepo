import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def residual_block(previous_block):
  #adding 2 convolutional layers (64,3,same padding) w/ relu fx
  #and then add thme to the previous_block output
  x = layers.Conv2D(64,3, activation='relu', padding='same')(previous_block)
  x = layers.Conv2D(64,3,activation='relu', padding='same')(x)
  block_output = layers.add([x, previous_block])
  return block_output

inputs = keras.Input(shape=(32,32,3), name='img')
x = layers.Conv2D(32,3,activation='relu')(inputs)
x = layers.Conv2D(64,3,activation='relu')(x)
block_1_output = layers.MaxPooling2D(2)(x)

#add 2 conv layers (64,3, same pad) w/ relu (x3 times)
block_2_output = residual_block(block_1_output)
block_3_output = residual_block(block_2_output)
block_4_output = residual_block(block_3_output)

x = layers.Conv2D(64,3,activation='relu')(block_4_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(40)(x)

model = keras.Model(inputs, outputs, name='aircraft_resnet')
model.summary()


#model vidualization
keras.utils.plot_model(model, 'aircraft_model.png')

#load the aircrafts dataset from kaggle
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/drive/MyDrive/Kaggle"

%cd /content/drive/MyDrive/Kaggle

!kaggle datasets download -d a2015003713/militaryaircraftdetectiondataset

%ls

!unzip \militaryaircraftdetectiondataset.zip  -d \militaryaircraftdetectiondataset && rm *.zip  

#create A/Cs labels
aircrafts = ["A10","A400M", "AG600","B1","B2","B52", "Be200", "C130","C17","C5", "E2","EF2000",
           "F117","F14","F15","F16","F18","F22","F35","F4",
           "JAS39","MQ9","Mig31","Mirage2000","RQ4","Rafale",
           "SR71","Su57","Tu160","Tu95","U2","US2", "V22","XB70","YF23","J20"]

#create a pd dataframe w/ images and its label in csv
path = "/content/drive/MyDrive/Kaggle/militaryaircraftdetectiondataset/annotated/"
images = glob.glob(path+ '*.jpg')
annot = []
for img in images: annot.append(img.replace('jpg','csv'))
df = pd.DataFrame({'image':images, 'annot':annot})
#preview
df.head()


#plot some dataset images to preview
def show_samples(n=5):
  img_index = list(np.round((np.random.random(n))*len(df)))
  fig, ax = plt.subplots(1,n,figsize=(10,10))
  i=0
  for idx in img_index:
    img = df.image.values[int(idx)]
    img = Image.open(img)
    ax[i].imshow(img)
    annot = pd.read_csv(df.annot.values[int(idx)])
    for j in range(len(annot)):
      x = annot.x.values[j] #anchor points: x,y
      y = annot.y.values[j]
      h = annot.h.values[j] #height
      w = annot.w.values[j] #width
      rect = patches.Rectangle((x,y,), w, h, linewidth=1, edgecolor='r', forecolor='none')
      #add patch to the axes
      ax[i].add_patch(rect)
    i +=1
  plt.show()
  return None    


show_samples(n=3)
