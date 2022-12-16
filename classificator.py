import numpy as np
import tensorflow as tf
from tensorflow import keras
import efficientnet as efn
from efficientnet.tfkeras import EfficientNetB6
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd

model = keras.models.load_model('./model/model_step4_sf.hdf5')

IMG_SIZE = 224

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor

def predict_image(img_tensor):
    classes = ['Лада Приора', 'Форд Фокус', 'Лада Самара 2114', 'Лада 2110', 'Лада 2107', 'Лада Нива', 'Лада Калина', 'Лада 2109', 'Фольксваген Пассат', 'Лада 21099']
    return classes[model.predict(img_tensor).argmax()]

if __name__ == "__main__":
    print(predict_image(load_image('data/train/5/1277.jpg')))

#print(model)