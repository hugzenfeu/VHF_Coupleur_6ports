import numpy as np
import cv2
import pylab as plt
from math import cos,sin,sqrt



def calculXY(L):
    point=[]
    for i in L:
        point.append([  (i[3]-i[4])/sqrt(2)/i[5]   ,  (i[3]+i[4]-i[2]-i[5])/2/sqrt(2)/i[5] ])
    return point


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


plt.figure()

#plt.plot(Xm,Ym,'go')
print(data[:10])


#print(data[0])
#point=calculXY(data)

print(len(X))
plt.plot(Xm,Ym,'go')
plt.plot(X,Y,'ro')
#plt.show()
point=[[Xm[i],Ym[i]]for i in range(len(data))]
pointPicked=[(len(data)*(i+1))//1000 for i in range(999)]

pts_src=np.array([point[i] for i in pointPicked])
pts_dst=np.array([[X[i],Y[i]]for i in pointPicked])

h, status = cv2.findHomography(pts_src, pts_dst)

for i in range(len(point)):
    point[i].append(1)

#print(point)

pointCori=[]

#print(h,type(h))

for i in range(len(point)):
    pointCori.append(np.dot(h,np.array(point[i])))
#print(pointCori)
XCor=[pointCori[i][0]/pointCori[i][2] for i in range(len(pointCori))]
YCor=[pointCori[i][1]/pointCori[i][2] for i in range(len(pointCori))]

plt.plot(XCor,YCor,'bo')

Err=[sqrt(Xm[i]**2+Ym[i]**2)-sqrt(X[i]**2+Y[i]**2) for i in range(len(X))]
ErrCor=[sqrt(XCor[i]**2+YCor[i]**2)-sqrt(X[i]**2+Y[i]**2) for i in range(len(X))]
plt.figure()
plt.plot(Err,'b',ErrCor,'r')
plt.show()
