import numpy as np


f=open("testInputbis.txt", "r")
lines =f.readlines()

"""
for x in lines:
    print(x)
"""
#print(len(lines[1:]))
#print(lines[46:])
content=lines
tamp=[]
data=np.zeros((len(content),4))
for i in range(12,len(content)-2):

    line=content[i]
    liste=line.replace('\n','').replace(' / ','	').split('	')
    #print(liste)
    if liste[0]!='' and liste[0]!='Y':
        tamp.append(liste)


#np.save("data_sim_1.npy",data)

for i in range(len(tamp)):
    data[i][0]=float(tamp[i][0])
    data[i][1]=float(tamp[i][1])
    data[i][2]=float(tamp[i][2])
    data[i][3]=float(tamp[i][3])

print(data[:100])
np.save("persp.npy",data)
