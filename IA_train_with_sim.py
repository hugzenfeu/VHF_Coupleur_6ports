

import tensorflow as tf
import pylab as plt
from math import cos,sin,sqrt
##test tensorboard
from time import time
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
from datetime import datetime


debut=time()
data=np.load('data_sim_1.npy')

np.random.shuffle(data)
#print(data[0])
#for i in range(len(data)):
    #data[i][0]=(data[i][0]-0.55)/1.1
    #data[i][1]=data[i][1]/360-0.5
    #data[i][2]=data[i][2]-3
    #data[i][3]=data[i][3]-3
    #data[i][4]=data[i][4]-3
    #data[i][5]=data[i][5]-3

proportion=int(len(data))

train=data[:proportion]
test=data[proportion:]

x_train=train[:,2:]
y_train=train[:,:2]

x_test=test[:,2:]
y_test=test[:,:2]



model =tf.keras.models.Sequential()  # random architecture

#print(1)
#model.add(keras.layers.Flatten(input_shape=(20,)))  #useless after all
model.add(tf.keras.layers.Dense(4))
model.add(tf.keras.layers.Dense(64, activation='linear'))
#model.add(tf.keras.layers.Dropout(0.01))   # prevent overfitting lol value might be a little bit high

model.add(tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))

model.add(tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))

#print(2)
model.add(tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
#model.add(tf.keras.layers.Dropout(0.01))   # prevent overfitting lol value might be a little bit high

model.add(tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))

model.add(tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))

model.add(tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))

model.add(tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))

model.add(tf.keras.layers.Dense(8 , activation=tf.keras.layers.LeakyReLU(alpha=0.3)))


model.add(tf.keras.layers.Dense(2 , activation='linear'))




opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['accuracy'],
)


print_loss=[]

now = datetime.now() # current date and time
date_time = now.strftime("%m/%d/%Y-%H:%M:%S")
for i in range(10):
    a=time()
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y-%H:%M:%S")

    opt = tf.keras.optimizers.Adam(lr=1/(1000*(i+1)-900), beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)

    history=model.fit(x_train,
                y_train,
                epochs=1000 ,
                validation_data=(x_test, y_test),
                verbose=0,
                batch_size=900)
    #model.save_weights(f"models\\my_model_weights{date_time.replace(':','_').replace('/','_')}.h5")


    print_loss.append(history.history['loss'])
    print(date_time,":      ",time()-a,"    ",history.history['loss'][-1],"    ",i)







features = data[:,2:]

predictions = model.predict(features)

np.save("predictions_sim_1.npy",predictions)

Y=[]
X=[]


for x in data[:,:2]:
    X.append((x[0])*cos((x[1])/360*2*3.14))
    Y.append((x[0])*sin((x[1])/360*2*3.14))

Y1=[]
X1=[]


for x in predictions:
    X1.append((x[0])*cos((x[1])/360*2*3.14))
    Y1.append((x[0])*sin((x[1])/360*2*3.14))
plt.figure()
plt.plot(X,Y,'ro',X1,Y1,'bo')

finloss=[]
for x in print_loss:
    for y in x:
        finloss.append(y)

plt.figure()
plt.plot(finloss,'b')

print(time()-debut)

plt.show()
