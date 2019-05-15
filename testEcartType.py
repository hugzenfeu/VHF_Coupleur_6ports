

import tensorflow as tf
import pylab as plt
from math import cos,sin,sqrt
##test tensorboard
from time import time
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
from datetime import datetime





def variations(data):
    R=[[0,0] for i in range(len(data[0]))]
    for x in data:
        for i in range(len(data[0])):
            R[i][0]+=x[i]
    for i in range(len(data[0])):
        R[i][0]=R[i][0]/len(data)

    for x in data:
        for i in range(len(data[0])-1):#issue constant input bigger dataset needed
            R[i][1]+=(x[i]-R[i][0])**2
    for i in range(len(data[0])-1):#issue constant input bigger dataset needed
        R[i][1]=sqrt(R[i][1]/len(data))
    return R



data=np.load('data_sim_1.npy')
print(data[:,5])
np.random.shuffle(data)
R=variations(data)

print(R)

for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j]=(data[i][j]-R[j][0])

print(data[:,5])
R=variations(data)

print(R)

"""
#print(data[0])
for i in range(len(data)):
    data[i][0]=(data[i][0]-0.55)/1.1
    data[i][1]=data[i][1]/360-0.5



#ecart type x[0]=0.31622776601683783
#ecart type x[1]=0.2910730290330117
sum=0
moy=0
for x in data:
    sum+=x[1]
moy=moy/len(data)

for x in data:
    sum+=(x[1]-moy)**2
sum=sqrt(sum/len(data))
print(sum)
"""
