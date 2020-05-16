import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time


X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))
X = X/255.0

datagen = ImageDataGenerator()
it = datagen.flow(X, Y, batch_size = 32)


name = "Classification-{}".format(int(time.time()))
tb = TensorBoard(log_dir='logs/{}'.format(name))

model = Sequential()
model.add(Conv2D(128, (5,5), input_shape=X.shape[1:]))
#model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (5,5)))
#model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (5,5)))
#model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
# model.add(Dense(32))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X, Y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tb])
#model.fit(it, steps_per_epoch=len(X)//32, epochs=5)

model.save("{}.h5".format(name))