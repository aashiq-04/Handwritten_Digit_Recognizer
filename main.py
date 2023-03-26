import tensorflow as tf
from keras.layers import *
from keras import *
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt


def load_dataset(path="data/"):
    train = pd.read_csv(path+"train.csv").to_numpy()
    x = []
    y = []
    for i in train:
        y.append(i[0])
        temp = list(i)
        temp.remove(i[0])
        x.append(temp)
    x = np.array(x)
    y = np.array(y)
    return x, y


def get_model():
    model = Sequential()
    model.add(Input(shape=(784,)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(10, activation="linear"))
    model.summary()

    return model


def train(model=Sequential(), train_x=[], train_y=[]):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=10, verbose=2)


def get_max_position(x):
    mx = -100000
    pos = -1
    for idx, i in enumerate(x):
        if i > mx:
            mx = i
            pos = idx
    return pos


def show_image(x, model):
    idx = np.random.randint(0, 28000)
    y = model.predict(x)
    plt.imshow(x[idx].reshape((28, 28)))
    plt.xlabel(get_max_position(y[idx]))
    plt.show()


x, y = load_dataset()


model = get_model()

train(model, x, y)

show_image(x, model)
