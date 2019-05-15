import numpy as np
import tensorflow as tf

##test tensorboard
from time import time
from tensorflow.keras.callbacks import TensorBoard



data=np.load('data_sim_1.npy')


model =tf.keras.models.Sequential()  # random architecture

#print(1)
#model.add(keras.layers.Flatten(input_shape=(20,)))  #useless after all
model.add(tf.keras.layers.Dense(4))

#print(2)
model.add(tf.keras.layers.Dense(15, activation='linear'))
#model.add(tf.keras.layers.Dropout(0.01))   # prevent overfitting lol value might be a little bit high

model.add(tf.keras.layers.Dense(10, activation='linear'))

model.add(tf.keras.layers.Dense(2, activation='linear'))

model.model.load_weights('K:\\data\\current_projects\\projet_tech\\models\\my_model_weights04_30_2019-00_30_43.h5')

opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1.0e-5, amsgrad=False)


# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['accuracy'],
)


#test tensorboard
tensorboard = TensorBoard(log_dir="my-test-model/{}".format(int(time())))

model.fit(x_train,
          y_train,
          epochs=60000  ,
          validation_data=(x_test, y_test),
          verbose=1,
          callbacks=[tensorboard])
