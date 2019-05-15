import numpy as np
import pylab as plt
from math import cos,sin




predictions = np.load("predictions_sim_1.npy")
data = np.load("data_sim_1.npy")
Y=[]
X=[]

for x in predictions:
    X.append(x[0]*cos(x[1]/360))
    Y.append(x[0]*sin(x[1]/360))

#print(predictions)

#print(sum(data[:,2]),sum(data[:,3]),sum(data[:,4]),sum(data[:,5]))
print([1.6**i for i in range(30)])
