import numpy as np
from matplotlib import pyplot as plt
from math import pi, cos, sin

u=2.0     #x-position of the center
v=3.0     #y-position of the center
p=0.      #Origin xcoordinate
q=0.      #Origin xcoordinate
a=2.      #radius on the x-axis
b=2.45    #radius on the y-axis
t_rot=-63.435*pi/180 #rotation angle

t = np.linspace(-5, 5, 100)
Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
     #u,v removed to keep the same center location
R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
     #2-D rotation matrix

Ell_rot = np.zeros((2,Ell.shape[1]))
for i in range(Ell.shape[1]):
    Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

plt.scatter(2,3)
plt.plot( u+Ell_rot[0,:] , v+Ell_rot[1,:],label='Given Ellipse' )    # given ellipse
plt.plot( p+Ell_rot[0,:] , q+Ell_rot[1,:],label='Ellipse at Origin' )    # same ellipse at origin
plt.plot(u,v)
plt.plot(p,q)
plt.grid()
plt.legend(loc='best')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.show()