from random import *
import numpy as np
import tensorflow as tf
import pylab as plt
from math import cos,sin
from time import time
from datetime import datetime

#from tensorflow import keras

###
def calculSortie(L):
    S , i = L[0],L[1]
    a=[i[2],0]
    b=[i[2]*i[1],i[0]]
    return [
    (a[0]*S[0][0]+b[0]*S[3][0])**2+(a[1]*S[0][1]+b[1]*S[3][1])**2,
    (a[0]*S[1][0]+b[0]*S[5][0])**2+(a[1]*S[1][1]+b[1]*S[5][1])**2,
    (a[0]*S[2][0]+b[0]*S[6][0])**2+(a[1]*S[2][1]+b[1]*S[6][1])**2,
    (a[0]*S[3][0]+b[0]*S[7][0])**2+(a[1]*S[3][1]+b[1]*S[7][1])**2
    ]


# re  im
#Styp=np.array([1,0,0.7,1,])         entrer typical S param

Num=[1]
"""ds l'ordre S13,S14,S15,S16,S23,S24,S25,S26 """
def generSparam(rang,x): #create Sparam "around" the typical ones
    Styp=[[-0.5,0.0],[0.0,0.5],[-0.5,0.0],[0.0,0.5],[0.5,0.0],[0.0,0.5],[0.0,0.5],[-0.5,0.0]]
    num = Num[0]
    Num[0]+=1
    S=[]
    for j in range(8):
        S.append([Styp[j][0]+rang/x*num-rang/2,Styp[j][1]+rang/x*num-rang/2])

    """S=Styp+range/num*i+range/num*i*1j"""

    return S.copy()




def flatten(A):
    L=[]
    for i in A:
        for j in i:
            L.append(j)
    return L

def features2Array(x):
    X=np.zeros((20))
    for i in range(20):
        X[i]=x[i]
    return X

def labels2Array(x):
    X=np.zeros((4))
    for i in range(4):
        X[i]=x[i]
    return X

def generInput():#create random but likely inputs
    return [random()*20-10,random()*20-10,random()*20-10]  # rep phase, rap amplitude, puissance d'entree en racine

def generS():# create random S param
    return [[random()*2-1,random()*2-1] for i in range(8)]

def generDataset(len):
    Dataset=[]
    for x in range(len):
        S=generSparam(len,x+1)
        i=generInput()
        x=np.array(flatten(S)+calculSortie([S,i]))
        y=np.array(i)
        Dataset.append([x,y])
    return Dataset

def split(Dataset):
    X,Y=[],[]
    for i in Dataset:
        X.append(i[0])
        Y.append(i[1])
    return X,Y

trainlen=200000

train=generDataset(trainlen)
x_train, y_train = split(train)  ## .copy()  ??

x_train, y_train=np.array(x_train), np.array(y_train)

now = datetime.now() # current date and time
date_time = now.strftime("%m/%d/%Y-%H:%M:%S")


#np.save(f"x-train_len-{trainlen}_date-{date_time.replace(':','_').replace('/','_')}",x_train)
#np.save(f"y-train_len-{trainlen}_date-{date_time.replace(':','_').replace('/','_')}",y_train)



test=generDataset(200)
x_test, y_test = split(test)
x_test, y_test=np.array(x_test), np.array(y_test)



print(y_test.shape)
print(x_test.shape)


# model
a=time()
model =tf.keras.models.Sequential()  # random architecture

#print(1)
#model.add(keras.layers.Flatten(input_shape=(20,)))  #useless after all
model.add(tf.keras.layers.Dense(20))
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


model.add(tf.keras.layers.Dense(3 , activation='linear'))




opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['accuracy'],
)


print_loss=[]


for i in range(1):
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y-%H:%M:%S")

    opt = tf.keras.optimizers.Adam(lr=1/100/10**i, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)

    history=model.fit(x_train,
                y_train,
                epochs=1000 ,
                validation_data=(x_test, y_test),
                verbose=1,
                batch_size=20000)
    #model.save_weights(f"models\\my_model_weights{date_time.replace(':','_').replace('/','_')}.h5")


    print_loss.append(history.history['loss'])







features = x_test

predictions = model.predict(features)

np.save("predictions_sim_1.npy",predictions)

Y=[]
X=[]


for x in x_test:
    X.append((x[0]*1.1+0.55)*cos((x[1]+0.5)*2*3.14))
    Y.append((x[0]*1.1+0.55)*sin((x[1]+0.5)*2*3.14))

Y1=[]
X1=[]


for x in predictions:
    X1.append((x[0]*1.1+0.55)*cos((x[1]+0.5)*2*3.14))
    Y1.append((x[0]*1.1+0.55)*sin((x[1]+0.5)*2*3.14))
plt.figure()
plt.plot(X,Y,'ro',X1,Y1,'bo')

finloss=[]
for x in print_loss:
    for y in x:
        finloss.append(y)

plt.figure()
plt.plot(finloss,'b')
b=time()

print(time()-a)
plt.show()
