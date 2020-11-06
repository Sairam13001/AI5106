import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv
from numpy import linalg as LA
from numpy import zeros
from numpy import diag
from numpy import dot

#Creating matrices
m1 = np.array(([1,2,0]))
m2 = np.array(([0,2,1]))
A = np.vstack((m1,m2)).T
b=np.array([2,5,-2])

#Eigenvalue decompostion of A'A
Dv, Pv=LA.eig(A.T.dot(A))

#Eigenvalue decompostion of AA'
Du, Pu=LA.eig(A.dot(A.T))

#Singular values of A
Stemp=np.sqrt(Dv)

#QR Decomposition to get U and V
V, Rv=LA.qr(Pv)
U, Ru=LA.qr(Pu)

#SVD, A=USV
U_1, s, V_1=LA.svd(A)
Sigma = zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
B = U.dot(Sigma.dot(V_1))

print(V)
print(V_1)
print(U)
print(U_1)
print(s)
print(Stemp)
print(B)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')


ax.plot3D([3], [4], [-1], marker='o', label='P = (3, 4, -1)')
ax.text(3, 4, -1,'P')


# equation line parallel to given line
ax.plot3D([7, 3], [2,4], [3, -1], color='b')

#creating x,y for 3D plotting
xx, yy = np.meshgrid([2,8], range(6)) 
n1 = np.array([2,-1,2]).reshape((3,1))

A =  np.array([3,4,-1]).reshape((3,1))
c1 = 5
#corresponding z for planes
z1 = (c1-n1[0]*xx-n1[1]*yy)/(n1[2])

#plotting planes
Plane=ax.plot_surface(xx, yy, z1,label='Plane', color='r',alpha=0.5)
Plane._facecolors2d=Plane._facecolors3d
Plane._edgecolors2d=Plane._edgecolors3d
#plotting point
plt.legend(loc='best')
plt.show()