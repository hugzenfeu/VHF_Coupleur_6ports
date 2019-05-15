from random import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM
from tensorflow import keras

###
def calculSortie(L):
    S,a,b=L[0],L[1],L[2]
    return [[
    (a[0]*S[0][0]+b[0]*S[3][0])**2+(a[1]*S[0][1]+b[1]*S[3][1])**2,
    (a[0]*S[1][0]+b[0]*S[5][0])**2+(a[1]*S[1][1]+b[1]*S[5][1])**2,
    (a[0]*S[2][0]+b[0]*S[6][0])**2+(a[1]*S[2][1]+b[1]*S[6][1])**2,
    (a[0]*S[3][0]+b[0]*S[7][0])**2+(a[1]*S[3][1]+b[1]*S[7][1])**2
    ]]


# re  im
#Styp=np.array([1,0,0.7,1,])         entrer typical S param

"""

def generSparam(range, num):
    L=[]
    for i in range(num):
        S=np.copy(Styp)+range/num*i+range/num*i*1j
        L.append(S.copy())
    return L
#S13,S14,S15,S16,S23,S24,S25,S26"""


def flatten(A):
    L=[]
    for i in A:
        for j in i:
            L.append(j)
    return L


def generInput():
    return [random()*20-10,random()*20-10]

def generS():
    return [[random()*2-1,random()*2-1] for i in range(8)]

def generDataset(len):
    Dataset=[]
    for x in range(len):
        S=generS()
        a=generInput()
        b=generInput()
        x=flatten(S)+flatten(calculSortie([S,a,b]))
        y=a+b
        Dataset.append([x,y])
    return Dataset

def split(Dataset):
    X,Y=[],[]
    for i in Dataset:
        X.append([i[0]])
        Y.append([i[1]])
    return X,Y


train=generDataset(20000)
x_train, y_train = split(train)  ## .copy()  ??

x_train, y_train =np.array(x_train), np.array(y_train)

test=generDataset(20000)
x_test, y_test = split(test)
x_test, y_test =np.array(x_test), np.array(y_test)

print(type(x_test[0][0]))
