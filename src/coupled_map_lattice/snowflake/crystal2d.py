#!/opt/local/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 18:49:16 2019

@author: yanagita
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:54:39 2019

@author: yanagita
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
#plt.rcParams['animation.ffmpeg_path'] = '/opt/local/bin/ffmpeg'

#matplotlib.use('TKAgg')
##############################################
def init():
  global u1,u2,x1,x2

  u1[0:,0:]=T0
  x1[int((L)/2),int((L)/2)]=1.0
  u1[int((L)/2),int((L)/2)]=Tc

  np.copyto(u2,u1)
  np.copyto(x2,x1)
  return line,
  
def bound():
  u1[int((L)/2),int((L)/2)]=Tc
##############################################
def update(time):
  global u1,u2,x1,x2
  for n in range(0,t_step):

    for i in range(1,L-1):
      for j in range(1,L-1):
        u2[i,j]=u1[i,j]+D*(u1[i+1,j]+u1[i-1,j]+u1[i,j+1]+u1[i,j-1]-4*u1[i,j])
    np.copyto(u1,u2)
    
    bound()
  
    for i in range(1,L-1):
      for j in range(1,L-1):
        if(u1[i,j]<Tc and (x1[i+1,j]>=1.0 or x1[i,j+1]>=1.0 or x1[i-1,j]>=1.0 or x1[i,j-1]>=1.0)):
          x2[i,j]=x1[i,j]+C1*(Tc-u1[i,j])
          u2[i,j]=u1[i,j]+C2*(Tc-u1[i,j])
    np.copyto(u1,u2)
    np.copyto(x1,x2)
    bound()

  fig.clf()
  cont=plt.contourf(x_axis,y_axis,x1,np.linspace(-0.5,1.5,num=10))
  contbar=plt.colorbar(cont)
  plt.xlabel('x [a.u.]')
  plt.ylabel('y [a.u.]')
  plt.title('CML model for crystal growth')
  print(time,np.sum(x1))
  return line,

##############################################
Tc=1.0
T0=-0.5
L=100
D=0.2
C1=0.3
C2=0.95
line=[]

t_step=2
x_axis=range(L)
y_axis=range(L)

u1=np.zeros((L,L))
x1=np.zeros((L,L))
u2=np.zeros((L,L))
x2=np.zeros((L,L))

#init()

  
fig=plt.figure()
ani=anim.FuncAnimation(fig,update,init_func=init,frames=500,interval=1)
ani.save("src/coupled_map_lattice/snowflake/plots/crystal2d-1.mp4", fps=10)

#plt.show()

