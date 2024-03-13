import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

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
# def update(time):
# 	global u1, u2, x1, x2
# 	for n in range(0, t_step):
# 		# Vectorized diffusion calculation using np.roll
# 		u2 = u1 + D * (np.roll(u1, -1, axis=0) + np.roll(u1, 1, axis=0) + np.roll(u1, -1, axis=1) + np.roll(u1, 1, axis=1) - 4 * u1)
# 		u1, u2 = u2, u1

# 		bound()

# 		# Vectorized freezing condition
# 		mask = (x1 < 1) & ((np.roll(x1, -1, axis=0) >= 1.0) | (np.roll(x1, 1, axis=0) >= 1.0) | (np.roll(x1, -1, axis=1) >= 1.0) | (np.roll(x1, 1, axis=1) >= 1.0))
# 		x2[mask] = x1[mask] + C1 * (u1[mask] - Tc)
# 		u2[mask] = u1[mask] - C2 * (u1[mask] - Tc)
# 		u1, u2 = u2, u1
# 		x1, x2 = x2, x1

# 		bound()

# 	fig.clf()
# 	cont=plt.contourf(x_axis,y_axis,x1,np.linspace(-0.5,1.5,num=10))
# 	contbar=plt.colorbar(cont)
# 	plt.xlabel('x [a.u.]')
# 	plt.ylabel('y [a.u.]')
# 	plt.title('CML model for crystal growth')
# 	print(time,np.sum(x1))
# 	return line,
  
# def update(time):
#   global u1,u2,x1,x2
#   for n in range(0,t_step):

#     for i in range(1,L-1):
#       for j in range(1,L-1):
#         u2[i,j]=u1[i,j]+D*(u1[i+1,j]+u1[i-1,j]+u1[i,j+1]+u1[i,j-1]-4*u1[i,j])
#     np.copyto(u1,u2)
    
#     bound()
  
#     for i in range(1,L-1):
#       for j in range(1,L-1):
#         if(x1[i,j]<0 and (x1[i+1,j]>=1.0 or x1[i,j+1]>=1.0 or x1[i-1,j]>=1.0 or x1[i,j-1]>=1.0)):
#           x2[i,j]=x1[i,j]+C1*(u1[i,j]-Tc)
#           u2[i,j]=u1[i,j]-C2*(u1[i,j]-Tc)
#     np.copyto(u1,u2)
#     np.copyto(x1,x2)
#     bound()

#   fig.clf()
#   cont=plt.contourf(x_axis,y_axis,x1,np.linspace(-0.5,1.5,num=10))
#   contbar=plt.colorbar(cont)
#   plt.xlabel('x [a.u.]')
#   plt.ylabel('y [a.u.]')
#   plt.title('CML model for crystal growth')
#   print(time,np.sum(x1))
#   return line,


##############################################
Tc=0
T0=0.01
L=300
D=0.15
C1=1
C2=1
line=[]

t_step=20
x_axis=range(L)
y_axis=range(L)

u1=np.zeros((L,L))
x1=np.zeros((L,L))
u2=np.zeros((L,L))
x2=np.zeros((L,L))

#init()

  
fig=plt.figure()
ani=anim.FuncAnimation(fig,update,init_func=init,frames=500,interval=1)
ani.save(f"src/coupled_map_lattice/snowflake/plots/crystal2d-1.mp4", fps=10)

#plt.show()

