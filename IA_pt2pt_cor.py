

import tensorflow as tf
import pylab as plt
from math import cos,sin,sqrt
from time import time
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from datetime import datetime


#calcul moyenne ecart type
def variations(data):
    R=[[0,0] for i in range(len(data[0]))]
    for x in data:
        for i in range(len(data[0])):#issue constant input bigger dataset needed
            R[i][0]+=x[i]
    for i in range(len(data[0])):#issue constant input bigger dataset needed
        R[i][0]=R[i][0]/len(data)

    for x in data:
        for i in range(len(data[0])):#issue constant input bigger dataset needed
            R[i][1]+=(x[i]-R[i][0])**2
    for i in range(len(data[0])):#issue constant input bigger dataset needed
        R[i][1]=sqrt(R[i][1]/len(data))

    return R
"""
    for i in range(len(R)):
        for j in range(len(R[0])):
            R[i][j]=0
"""


#mesure temps d'execution
debut=time()

data=np.load('persp.npy')
#data=data[:500]

Y=[]
X=[]
Xm=[]
Ym=[]

for x in data:
    X.append((x[0])*cos((x[1])/360*2*3.14))
    Y.append((x[0])*sin((x[1])/360*2*3.14))
    Xm.append((x[2])*cos((x[3])/360*2*3.14))
    Ym.append((x[2])*sin((x[3])/360*2*3.14))

for i in range(len(data)):

    data[i][0]=X[i]
    data[i][1]=Y[i]
    data[i][2]=Xm[i]
    data[i][3]=Ym[i]


#melange des donnés pour eviter l'overfitting
np.random.shuffle(data)
#print(data[0])

#Normalization des données
R=variations(data)
for i in range(len(data)):
    for j in range(len(data[0])):#issue constant input bigger dataset needed
        data[i][j]=(data[i][j]-R[j][0])/R[j][1]

print(variations(data))

#separation training set validation set
proportion=int(len(data)*0.9)

train=data[:proportion]
test=data[proportion:]

x_train=train[:,2:4]
y_train=train[:,:2]

x_test=test[:,2:4]
y_test=test[:,:2]


#architecture du modele
model =tf.keras.models.Sequential()  # random architecture

#print(1)
#model.add(keras.layers.Flatten(input_shape=(20,)))  #useless after all
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Dense(64, activation='linear'))
#model.add(tf.keras.layers.Dropout(0.01))   # prevent overfitting lol value might be a little bit high

model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.01))   # prevent overfitting lol value might be a little bit high


model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(32, activation='relu'))

model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(8 , activation='relu'))


model.add(tf.keras.layers.Dense(2 , activation='linear'))



#parametres d'apprentissage
opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['accuracy'],
)

#memoire loss
print_loss=[]

#création clef primaire pour nouveaux modeles
now = datetime.now() # current date and time
date_time = now.strftime("%m/%d/%Y-%H:%M:%S")

for i in range(1):
    #mesure temps d'execution
    a=time()


    #opt = tf.keras.optimizers.Adam(lr=1/(10000*(i+1)-5000), beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)# sert a rien
    #entrainement du réseau
    history=model.fit(x_train,
                y_train,
                epochs=1000 ,
                validation_data=(x_test, y_test),
                verbose=1,
                batch_size=900)



    print_loss.append(history.history['loss'])
    print(date_time,":      ",time()-a,"    ",history.history['loss'][-1],"    ",i)

#enregistrement etat du modèle
model.save_weights(f"models\\bon_modele{date_time.replace(':','_').replace('/','_')}.h5")



#prediction du modele
features = data[:,2:4]

predictions = model.predict(features)

np.save("predictions_sim_1.npy",predictions)


#affichage
Y=[]
X=[]


for x in data[:,:2]:
    #X.append((x[0]*R[0][1]+R[0][0])*cos((x[1]*R[1][1]+R[1][0])/360*2*3.14))
    #Y.append((x[0]*R[0][1]+R[0][0])*sin((x[1]*R[1][1]+R[1][0])/360*2*3.14))
    X.append(x[0])
    Y.append(x[1])
Y1=[]
X1=[]


for x in predictions:
    #X1.append((x[0]*R[0][1]+R[0][0])*cos((x[1]*R[1][1]+R[1][0])/360*2*3.14))
    #Y1.append((x[0]*R[0][1]+R[0][0])*sin((x[1]*R[1][1]+R[1][0])/360*2*3.14))
    X1.append(x[0])
    Y1.append(x[1])

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
