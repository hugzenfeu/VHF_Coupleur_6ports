

import tensorflow as tf
import pylab as plt
from math import cos,sin,sqrt
##test tensorboard
from time import time
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
from datetime import datetime


def moyenneGlissante(L):
    R=[]
    halfsize=100
    for i in range(halfsize,len(L)-halfsize):
        R.append(sum(L[i-halfsize:i+halfsize])/halfsize/2)
    return R


def variations(data):
    R=[[0,0] for i in range(len(data[0]))]
    for x in data:
        for i in range(len(data[0])-1):#issue constant input bigger dataset needed
            R[i][0]+=x[i]
    for i in range(len(data[0])-1):#issue constant input bigger dataset needed
        R[i][0]=R[i][0]/len(data)

    for x in data:
        for i in range(len(data[0])-1):#issue constant input bigger dataset needed
            R[i][1]+=(x[i]-R[i][0])**2
    for i in range(len(data[0])-1):#issue constant input bigger dataset needed
        R[i][1]=sqrt(R[i][1]/len(data))
    return R



debut=time()



data=np.load('data_sim_1.npy')

np.random.shuffle(data)
#print(data[0])

R=variations(data)
for i in range(len(data)):
    for j in range(len(data[0])-1):
        data[i][j]=(data[i][j]-R[j][0])/R[j][1]

print(variations(data))

proportion=int(len(data))

train=data[:proportion]
test=data[proportion:]

x_train=train[:,2:5]
y_train=train[:,:2]

x_test=test[:,2:5]
y_test=test[:,:2]




model =tf.keras.models.Sequential()  # random architecture

#print(1)
#model.add(keras.layers.Flatten(input_shape=(20,)))  #useless after all
model.add(tf.keras.layers.Dense(3,input_shape=(3,)))
model.add(tf.keras.layers.Dense(64, activation='linear'))
#model.add(tf.keras.layers.Dropout(0.01))   # prevent overfitting lol value might be a little bit high

model.add(tf.keras.layers.Dense(128, activation='softsign'))
#model.add(tf.keras.layers.Dropout(0.01))   # prevent overfitting lol value might be a little bit high


model.add(tf.keras.layers.Dense(64, activation='softsign'))

model.add(tf.keras.layers.Dense(32, activation='softsign'))

model.add(tf.keras.layers.Dense(16, activation='softsign'))

model.add(tf.keras.layers.Dense(8 , activation='softsign'))


model.add(tf.keras.layers.Dense(2 , activation='linear'))


model_name="models\\my_model_weights_var_cor_lower_dim05_07_2019-15_55_39.h5"

model.load_weights(model_name)


opt = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['accuracy'],
)





print_loss=[]


for i in range(1):
    a=time()


    #opt = tf.keras.optimizers.Adam(lr=0.000000000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)#doesn't do anything

    history=model.fit(x_train,
                y_train,
                epochs=1000 ,
                validation_data=(x_test, y_test),
                verbose=0,
                batch_size=250)

    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y-%H:%M:%S")

    print_loss.append(history.history['loss'])
    model.save_weights(model_name)
    print(date_time,":      ",int(time()-a),"    ",sum(history.history['loss'])/len(history.history['loss']),"    ",i)
#model.save_weights(f"models\\my_model_weights_var_cor05_04_2019-18_03_55.h5")
features = data[:,2:5]

predictions = model.predict(features)

np.save("predictions_sim_1.npy",predictions)

Y=[]
X=[]


for x in data[:,:2]:
    X.append((x[0]*R[0][1]+R[0][0])*cos((x[1]*R[1][1]+R[1][0])/360*2*3.14))
    Y.append((x[0]*R[0][1]+R[0][0])*sin((x[1]*R[1][1]+R[1][0])/360*2*3.14))

Y1=[]
X1=[]


for x in predictions:
    X1.append((x[0]*R[0][1]+R[0][0])*cos((x[1]*R[1][1]+R[1][0])/360*2*3.14))
    Y1.append((x[0]*R[0][1]+R[0][0])*sin((x[1]*R[1][1]+R[1][0])/360*2*3.14))
plt.figure()
plt.plot(X,Y,'ro',X1,Y1,'bo')

finloss=[]
for x in print_loss:
    for y in x:
        finloss.append(y)

plt.figure()
plt.plot(moyenneGlissante(finloss),'b')

plt.figure()
plt.plot(finloss,'b')
print(time()-debut)

plt.show()
