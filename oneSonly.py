from random import *
import numpy as np
import tensorflow as tf

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
        Styp=[[-0.5,0.0],[0.0,0.5],[-0.5,0.0],[0.0,0.5],[0.5,0.0],[0.0,0.5],[0.0,0.5],[-0.5,0.0]]
        i=generInput()
        x=np.array(calculSortie([Styp,i]))
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



test=generDataset(1000)
x_test, y_test = split(test)
x_test, y_test=np.array(x_test), np.array(y_test)

"""
print(x_test.shape)
#print(y_test.shape)
inputs = tf.keras.Input(shape=(4,), name='digits')
x = tf.keras.layers.Dense(4, activation='relu', name='dense_1')(inputs)
x = tf.keras.layers.Dense(40, activation='relu', name='dense_2')(x)
x = tf.keras.layers.Dense(20, activation='relu', name='dense_3')(x)
outputs = tf.keras.layers.Dense(3, activation='linear', name='predictions')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop())
history = model.fit(x_test, y_test,
                    batch_size=64,
                    epochs=3)"""

"""test saving doesn't work yet"""
# model

model =tf.keras.models.Sequential()  # random architecture

#print(1)
#model.add(keras.layers.Flatten(input_shape=(20,)))  #useless after all
model.add(tf.keras.layers.Dense(4))

#print(2)
model.add(tf.keras.layers.Dense(40, activation='linear'))
#model.add(tf.keras.layers.Dropout(0.01))   # prevent overfitting lol value might be a little bit high

model.add(tf.keras.layers.Dense(20))

model.add(tf.keras.layers.Dense(3, activation='linear'))


opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)



# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['accuracy'],
)



model.fit(x_train,
          y_train,
          epochs=1500  ,
          validation_data=(x_test, y_test))
model.save_weights(f"models\\my_model_weights{date_time.replace(':','_').replace('/','_')}.h5")
#print(model.weights[0])
#model.save('K:\\data\\current_projects\\projet_tech\\my-test-model.h5')
